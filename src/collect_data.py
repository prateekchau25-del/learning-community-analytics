"""
Collect a learning-community dataset from the Stack Exchange API.

Downloads questions, answers, and comments from one Stack Exchange site and
saves them as CSV files that the rest of the project (topic modeling,
network analysis, diffusion) builds on.

Usage:
    python src/collect_data.py --site datascience --max-questions 2000

Without an API key you get 300 requests/day, which is enough for ~5,000
questions. Register a free key at https://stackapps.com/apps/oauth/register
to get 10,000 requests/day, then pass it with --key.
"""

import argparse
import html
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://api.stackexchange.com/2.3"
# Built-in filter that includes the post body text.
FILTER = "withbody"
MAX_PAGE_WITHOUT_KEY = 25
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

class StackExchangeClient:
    def __init__(self, site, key=None):
        self.site = site
        self.key = key
        self.quota_remaining = None

    def get(self, endpoint, **params):
        """Yield every item from a paginated endpoint."""
        params.update(site=self.site, filter=FILTER, pagesize=100, page=1)
        if self.key:
            params["key"] = self.key
        # Without a key the API refuses pages above 25.
        max_page = None if self.key else MAX_PAGE_WITHOUT_KEY
        while max_page is None or params["page"] <= max_page:
            resp = requests.get(f"{API}/{endpoint}", params=params, timeout=30)
            data = resp.json()
            if "error_id" in data:
                raise RuntimeError(f"API error {data['error_id']}: {data.get('error_message')}")
            self.quota_remaining = data.get("quota_remaining")
            yield from data.get("items", [])
            # The API asks clients to wait when it sends a backoff value.
            if data.get("backoff"):
                time.sleep(data["backoff"])
            if not data.get("has_more"):
                break
            params["page"] += 1
            time.sleep(0.1)


def owner_fields(item):
    owner = item.get("owner", {})
    return {
        "user_id": owner.get("user_id"),
        "user_name": html.unescape(owner.get("display_name", "")),
        "user_reputation": owner.get("reputation"),
    }


def fetch_questions(client, max_questions):
    """
    Newest questions first. The API only serves 25 pages per query without a
    key, so after each 25-page window we continue from the oldest date seen.
    """
    rows = {}
    todate = None
    while len(rows) < max_questions:
        params = {"order": "desc", "sort": "creation"}
        if todate:
            params["todate"] = todate
        before = len(rows)
        for q in client.get("questions", **params):
            rows[q["question_id"]] = {
                "post_id": q["question_id"],
                "title": html.unescape(q.get("title", "")),
                "body": q.get("body", ""),
                "tags": "|".join(q.get("tags", [])),
                "score": q.get("score"),
                "view_count": q.get("view_count"),
                "answer_count": q.get("answer_count"),
                "is_answered": q.get("is_answered"),
                "accepted_answer_id": q.get("accepted_answer_id"),
                "created_at": q.get("creation_date"),
                **owner_fields(q),
            }
            if len(rows) >= max_questions:
                break
            if len(rows) % 500 == 0:
                print(f"  questions: {len(rows)} (quota left: {client.quota_remaining})")
        if len(rows) == before:
            break  # no older questions left
        todate = min(r["created_at"] for r in rows.values())
    return pd.DataFrame(list(rows.values()))


def fetch_by_ids(client, endpoint_template, ids, parse):
    """Call an '{ids}' endpoint in batches of 100 ids (the API maximum)."""
    rows = []
    ids = [str(i) for i in ids]
    for start in range(0, len(ids), 100):
        batch = ";".join(ids[start:start + 100])
        rows.extend(parse(item) for item in client.get(endpoint_template.format(ids=batch)))
        print(f"  {endpoint_template.split('/')[-1]}: {min(start + 100, len(ids))}/{len(ids)} parents "
              f"(quota left: {client.quota_remaining})")
    return pd.DataFrame(rows)


def parse_answer(a):
    return {
        "post_id": a["answer_id"],
        "question_id": a["question_id"],
        "body": a.get("body", ""),
        "score": a.get("score"),
        "is_accepted": a.get("is_accepted"),
        "created_at": a.get("creation_date"),
        **owner_fields(a),
    }


def parse_comment(c):
    reply_to = c.get("reply_to_user", {})
    return {
        "comment_id": c["comment_id"],
        "post_id": c["post_id"],
        "body": c.get("body", ""),
        "score": c.get("score"),
        "reply_to_user_id": reply_to.get("user_id"),
        "created_at": c.get("creation_date"),
        **owner_fields(c),
    }


def build_interactions(questions, answers, comments):
    """
    One row per 'source user helped/replied to target user' event.
    This is the edge list for the knowledge network.
    """
    post_owner = pd.concat([
        questions[["post_id", "user_id"]],
        answers[["post_id", "user_id"]],
    ]).dropna().drop_duplicates("post_id").set_index("post_id")["user_id"]
    q_owner = questions.set_index("post_id")["user_id"]

    edges = []
    for a in answers.itertuples():
        edges.append((a.user_id, q_owner.get(a.question_id), "answer", a.post_id, a.question_id, a.created_at))
    for c in comments.itertuples():
        target = c.reply_to_user_id if pd.notna(c.reply_to_user_id) else post_owner.get(c.post_id)
        edges.append((c.user_id, target, "comment", c.comment_id, c.post_id, c.created_at))

    df = pd.DataFrame(edges, columns=["source_user", "target_user", "type", "id", "parent_post_id", "created_at"])
    df = df.dropna(subset=["source_user", "target_user"])
    return df[df.source_user != df.target_user]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--site", default="datascience", help="Stack Exchange site, e.g. datascience, ai, stats, cs50")
    parser.add_argument("--max-questions", type=int, default=2000)
    parser.add_argument("--key", default=None, help="Optional Stack Apps API key for a higher quota")
    args = parser.parse_args()

    out = OUT_DIR / args.site
    out.mkdir(parents=True, exist_ok=True)
    client = StackExchangeClient(args.site, args.key)

    print(f"Fetching up to {args.max_questions} questions from {args.site}.stackexchange.com ...")
    questions = fetch_questions(client, args.max_questions)
    questions.to_csv(out / "questions.csv", index=False)
    print(f"Saved {len(questions)} questions")

    print("Fetching answers ...")
    answers = fetch_by_ids(client, "questions/{ids}/answers", questions.post_id, parse_answer)
    answers.to_csv(out / "answers.csv", index=False)
    print(f"Saved {len(answers)} answers")

    print("Fetching comments ...")
    all_post_ids = pd.concat([questions.post_id, answers.post_id])
    comments = fetch_by_ids(client, "posts/{ids}/comments", all_post_ids, parse_comment)
    comments.to_csv(out / "comments.csv", index=False)
    print(f"Saved {len(comments)} comments")

    interactions = build_interactions(questions, answers, comments)
    interactions.to_csv(out / "interactions.csv", index=False)

    users = pd.concat([questions, answers, comments])[["user_id", "user_name", "user_reputation"]]
    users = users.dropna(subset=["user_id"]).sort_values("user_reputation").drop_duplicates("user_id", keep="last")
    users.to_csv(out / "users.csv", index=False)

    print(f"\nDone. Files in {out}")
    print(f"  questions={len(questions)} answers={len(answers)} comments={len(comments)} "
          f"users={len(users)} interactions={len(interactions)}")
    print(f"  API quota remaining today: {client.quota_remaining}")


if __name__ == "__main__":
    main()
