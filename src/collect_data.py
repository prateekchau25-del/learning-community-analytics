"""
Step 1: collect a learning-community dataset from the Stack Exchange API.

One request returns 100 questions together with all their answers and comments
(a custom API filter), so 10,000 questions cost only about 100 requests.
Downloaded pages are saved as they arrive, so if the daily quota runs out the
next run continues where the last one stopped.

Usage:
    python src/collect_data.py --site datascience --max-questions 10000
    python src/collect_data.py --site ai --max-questions 10000

Without an API key you get 300 requests/day. With a free key from
https://stackapps.com/apps/oauth/register you get 10,000/day: put it in .env as
SE_API_KEY=... or pass --key.
"""

import argparse
import html
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

API = "https://api.stackexchange.com/2.3"
# Custom filter: question/answer/comment bodies, nested answers and comments,
# accepted_answer_id and the owner's network-wide account_id.
FILTER = "!.JVKLOvYtON4yuITFg325HDMLJ.k1"
MAX_PAGE_WITHOUT_KEY = 25
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
load_dotenv(PROJECT_ROOT / ".env")


class QuotaExhausted(Exception):
    pass


class StackExchangeClient:
    def __init__(self, site, key=None):
        self.site, self.key = site, key
        self.quota_remaining = None

    @staticmethod
    def _get_with_retry(url, params, attempts=5):
        """Retry short network failures (timeouts, dropped connections) with a growing wait."""
        for attempt in range(1, attempts + 1):
            try:
                return requests.get(url, params=params, timeout=60).json()
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt == attempts:
                    raise
                wait = 10 * attempt
                print(f"  network problem ({type(exc).__name__}); retrying in {wait}s ...")
                time.sleep(wait)

    def pages(self, endpoint, **params):
        """Yield (page_number, items) for a paginated endpoint."""
        params.update(site=self.site, filter=FILTER, pagesize=100, page=1)
        if self.key:
            params["key"] = self.key
        max_page = None if self.key else MAX_PAGE_WITHOUT_KEY
        while max_page is None or params["page"] <= max_page:
            data = self._get_with_retry(f"{API}/{endpoint}", params)
            if "error_id" in data:
                if data["error_id"] == 502 or "quota" in str(data.get("error_message", "")).lower():
                    raise QuotaExhausted(data.get("error_message"))
                raise RuntimeError(f"API error {data['error_id']}: {data.get('error_message')}")
            self.quota_remaining = data.get("quota_remaining")
            yield params["page"], data.get("items", [])
            if self.quota_remaining is not None and self.quota_remaining <= 1:
                raise QuotaExhausted("daily quota used up")
            if data.get("backoff"):
                time.sleep(data["backoff"])
            if not data.get("has_more"):
                return
            params["page"] += 1
            time.sleep(0.2)


def download(client, out_dir: Path, max_questions: int) -> list[dict]:
    """Newest questions first; resumes from questions.jsonl if it already exists."""
    store = out_dir / "questions.jsonl"
    seen = {}
    if store.exists():
        bad = 0
        for line in store.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip("\x00 \t")
            if not line:
                continue
            try:
                q = json.loads(line)
            except json.JSONDecodeError:
                bad += 1  # a line cut off when a previous run was interrupted
                continue
            seen[q["question_id"]] = q
        if bad:
            # Rewrite the file without the damaged lines before appending to it.
            store.write_text("".join(json.dumps(q) + "\n" for q in seen.values()), encoding="utf-8")
            print(f"  removed {bad} damaged line(s) left by an interrupted run")
        print(f"  resuming: {len(seen)} questions already downloaded")
    todate = min((q["creation_date"] for q in seen.values()), default=None)
    with store.open("a", encoding="utf-8") as f:
        try:
            while len(seen) < max_questions:
                params = {"order": "desc", "sort": "creation"}
                if todate:
                    params["todate"] = todate
                before = len(seen)
                for page, items in client.pages("questions", **params):
                    for q in items:
                        if q["question_id"] not in seen:
                            seen[q["question_id"]] = q
                            f.write(json.dumps(q) + "\n")
                    f.flush()
                    print(f"  {len(seen):>6} questions (quota left: {client.quota_remaining})")
                    if len(seen) >= max_questions:
                        break
                if len(seen) == before:
                    break  # reached the oldest question on the site
                todate = min(q["creation_date"] for q in seen.values())
        except QuotaExhausted as exc:
            print(f"\n  Stopped: {exc}. Progress is saved; run the same command again later to continue.")
    return list(seen.values())


def owner(item):
    o = item.get("owner", {})
    return {"user_id": o.get("user_id"), "account_id": o.get("account_id"),
            "user_name": html.unescape(o.get("display_name", "")), "user_reputation": o.get("reputation")}


def flatten(raw: list[dict]):
    qs, ans, coms = [], [], []
    for q in raw:
        qs.append({
            "post_id": q["question_id"], "title": html.unescape(q.get("title", "")), "body": q.get("body", ""),
            "tags": "|".join(q.get("tags", [])), "score": q.get("score"), "view_count": q.get("view_count"),
            "answer_count": q.get("answer_count"), "is_answered": q.get("is_answered"),
            "accepted_answer_id": q.get("accepted_answer_id"), "created_at": q.get("creation_date"), **owner(q)})
        for a in q.get("answers", []):
            ans.append({"post_id": a["answer_id"], "question_id": q["question_id"], "body": a.get("body", ""),
                        "score": a.get("score"), "is_accepted": a.get("is_accepted"),
                        "created_at": a.get("creation_date"), **owner(a)})
        for p in [q] + q.get("answers", []):
            for c in p.get("comments", []):
                coms.append({"comment_id": c["comment_id"], "post_id": c["post_id"], "body": c.get("body", ""),
                             "score": c.get("score"), "reply_to_user_id": c.get("reply_to_user", {}).get("user_id"),
                             "created_at": c.get("creation_date"), **owner(c)})
    return pd.DataFrame(qs), pd.DataFrame(ans), pd.DataFrame(coms)


def build_interactions(questions, answers, comments):
    """One row per 'source user helped / replied to target user' event: the knowledge-network edge list."""
    post_owner = pd.concat([questions[["post_id", "user_id"]], answers[["post_id", "user_id"]]]) \
        .dropna().drop_duplicates("post_id").set_index("post_id")["user_id"]
    q_owner = questions.set_index("post_id")["user_id"]
    edges = [(a.user_id, q_owner.get(a.question_id), "answer", a.post_id, a.question_id, a.created_at)
             for a in answers.itertuples()]
    for c in comments.itertuples():
        target = c.reply_to_user_id if pd.notna(c.reply_to_user_id) else post_owner.get(c.post_id)
        edges.append((c.user_id, target, "comment", c.comment_id, c.post_id, c.created_at))
    df = pd.DataFrame(edges, columns=["source_user", "target_user", "type", "id", "parent_post_id", "created_at"])
    df = df.dropna(subset=["source_user", "target_user"])
    return df[df.source_user != df.target_user]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--site", default="datascience", help="Stack Exchange site, e.g. datascience, ai")
    parser.add_argument("--max-questions", type=int, default=10000)
    parser.add_argument("--key", default=os.getenv("SE_API_KEY"), help="Stack Apps API key (or SE_API_KEY in .env)")
    args = parser.parse_args()

    out = RAW_ROOT / args.site
    out.mkdir(parents=True, exist_ok=True)
    client = StackExchangeClient(args.site, args.key)
    print(f"Downloading up to {args.max_questions} questions from {args.site}.stackexchange.com "
          f"({'with' if args.key else 'without'} API key) ...")
    raw = download(client, out, args.max_questions)

    questions, answers, comments = flatten(raw)
    interactions = build_interactions(questions, answers, comments)
    users = pd.concat([questions, answers, comments])[["user_id", "account_id", "user_name", "user_reputation"]]
    users = users.dropna(subset=["user_id"]).sort_values("user_reputation").drop_duplicates("user_id", keep="last")
    for name, df in [("questions", questions), ("answers", answers), ("comments", comments),
                     ("users", users), ("interactions", interactions)]:
        df.to_csv(out / f"{name}.csv", index=False)

    print(f"\nDone. Files in {out}")
    print(f"  questions={len(questions)} answers={len(answers)} comments={len(comments)} "
          f"users={len(users)} interactions={len(interactions)}")
    print(f"  API quota remaining today: {client.quota_remaining}")


if __name__ == "__main__":
    main()
