-- Schema for the learning-community database.
-- Running this file recreates the tables from scratch (existing data is removed).

DROP TABLE IF EXISTS interactions, comments, answers, questions, users CASCADE;

CREATE TABLE users (
    user_id         BIGINT PRIMARY KEY,
    user_name       TEXT,
    user_reputation INTEGER
);

CREATE TABLE questions (
    post_id            BIGINT PRIMARY KEY,
    title              TEXT,
    body               TEXT,
    tags               TEXT,          -- pipe-separated, e.g. 'pset1|c|cs50'
    score              INTEGER,
    view_count         INTEGER,
    answer_count       INTEGER,
    is_answered        BOOLEAN,
    accepted_answer_id BIGINT,
    created_at         TIMESTAMP,
    user_id            BIGINT REFERENCES users(user_id)
);

CREATE TABLE answers (
    post_id     BIGINT PRIMARY KEY,
    question_id BIGINT NOT NULL REFERENCES questions(post_id),
    body        TEXT,
    score       INTEGER,
    is_accepted BOOLEAN,
    created_at  TIMESTAMP,
    user_id     BIGINT REFERENCES users(user_id)
);

-- A comment can belong to a question or an answer, so post_id has no foreign key.
CREATE TABLE comments (
    comment_id       BIGINT PRIMARY KEY,
    post_id          BIGINT NOT NULL,
    body             TEXT,
    score            INTEGER,
    reply_to_user_id BIGINT,
    created_at       TIMESTAMP,
    user_id          BIGINT REFERENCES users(user_id)
);

-- Edge list of the knowledge network: source_user helped / replied to target_user.
-- target_user may be someone with no posts in the sample, so no foreign key here.
CREATE TABLE interactions (
    interaction_id SERIAL PRIMARY KEY,
    source_user    BIGINT NOT NULL,
    target_user    BIGINT NOT NULL,
    type           TEXT NOT NULL CHECK (type IN ('answer', 'comment')),
    id             BIGINT NOT NULL,   -- answer or comment id
    parent_post_id BIGINT NOT NULL,
    created_at     TIMESTAMP
);

CREATE INDEX idx_questions_created ON questions(created_at);
CREATE INDEX idx_answers_question  ON answers(question_id);
CREATE INDEX idx_comments_post     ON comments(post_id);
CREATE INDEX idx_interactions_src  ON interactions(source_user);
CREATE INDEX idx_interactions_tgt  ON interactions(target_user);
