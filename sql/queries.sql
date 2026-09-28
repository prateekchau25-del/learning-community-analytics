-- Analysis queries to run in pgAdmin (Tools -> Query Tool).
-- Run each query on its own: select it and press F5.

-- 1. Dataset overview
SELECT
    (SELECT COUNT(*) FROM users)        AS users,
    (SELECT COUNT(*) FROM questions)    AS questions,
    (SELECT COUNT(*) FROM answers)      AS answers,
    (SELECT COUNT(*) FROM comments)     AS comments,
    (SELECT COUNT(*) FROM interactions) AS interactions;

-- 2. Top 10 helpers: most answers, and how many were accepted
SELECT u.user_name,
       u.user_reputation,
       COUNT(*)                                 AS answers_given,
       COUNT(*) FILTER (WHERE a.is_accepted)    AS accepted_answers
FROM answers a
JOIN users u ON u.user_id = a.user_id
GROUP BY u.user_id, u.user_name, u.user_reputation
ORDER BY answers_given DESC
LIMIT 10;

-- 3. Most discussed tags and how often their questions go unanswered
SELECT tag,
       COUNT(*)                                              AS questions,
       ROUND(100.0 * AVG(CASE WHEN q.is_answered THEN 0 ELSE 1 END), 1) AS pct_unanswered
FROM questions q,
     unnest(string_to_array(q.tags, '|')) AS tag
GROUP BY tag
HAVING COUNT(*) >= 10
ORDER BY questions DESC
LIMIT 20;

-- 4. Median hours until the first answer, per month
SELECT date_trunc('month', q.created_at)::date AS month,
       COUNT(*) AS answered_questions,
       ROUND((PERCENTILE_CONT(0.5) WITHIN GROUP (
           ORDER BY EXTRACT(EPOCH FROM fa.first_answer - q.created_at) / 3600))::numeric, 1)
           AS median_hours_to_first_answer
FROM questions q
JOIN (SELECT question_id, MIN(created_at) AS first_answer
      FROM answers GROUP BY question_id) fa ON fa.question_id = q.post_id
GROUP BY month
ORDER BY month;

-- 5. Knowledge diffusion: learners who first asked, and later became helpers
WITH first_question AS (
    SELECT user_id, MIN(created_at) AS first_asked FROM questions GROUP BY user_id
), first_answer AS (
    SELECT user_id, MIN(created_at) AS first_answered FROM answers GROUP BY user_id
)
SELECT u.user_name,
       fq.first_asked::date,
       fa.first_answered::date,
       fa.first_answered::date - fq.first_asked::date AS days_to_become_helper
FROM first_question fq
JOIN first_answer fa ON fa.user_id = fq.user_id AND fa.first_answered > fq.first_asked
JOIN users u ON u.user_id = fq.user_id
ORDER BY days_to_become_helper;

-- 6. Strongest helper -> learner links in the knowledge network
SELECT s.user_name AS helper,
       t.user_name AS learner,
       COUNT(*)    AS interactions
FROM interactions i
JOIN users s ON s.user_id = i.source_user
LEFT JOIN users t ON t.user_id = i.target_user
GROUP BY s.user_name, t.user_name
ORDER BY interactions DESC
LIMIT 15;
