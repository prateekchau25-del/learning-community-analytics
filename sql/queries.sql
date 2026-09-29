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

-- ===================================================================
-- Queries on the analysis results (tables created by load_to_postgres.py
-- after running src/run_pipeline.py)
-- ===================================================================

-- 7. Hardest topics: unanswered rate, waiting time and confusion
SELECT label, course_unit, n_questions,
       ROUND(pct_unanswered::numeric, 1)         AS pct_unanswered,
       ROUND(median_hours_to_answer::numeric, 1) AS median_hours_to_answer,
       ROUND(difficulty_index::numeric, 2)       AS difficulty_index
FROM topics
ORDER BY difficulty_index DESC;

-- 8. Top 3 experts for every topic (who answers the most questions in it)
WITH per_topic AS (
    SELECT qt.topic, a.user_id, COUNT(*) AS answers,
           ROW_NUMBER() OVER (PARTITION BY qt.topic ORDER BY COUNT(*) DESC) AS rnk
    FROM answers a
    JOIN question_topics qt ON qt.post_id = a.question_id
    GROUP BY qt.topic, a.user_id
)
SELECT t.label AS topic, u.user_name, p.answers
FROM per_topic p
JOIN topics t ON t.topic = p.topic
JOIN users u ON u.user_id = p.user_id
WHERE p.rnk <= 3
ORDER BY t.label, p.answers DESC;

-- 9. Network roles: how many users play each role, and how much help they give
SELECT role,
       COUNT(*)                                    AS users,
       SUM(out_strength)                           AS help_given,
       ROUND(100.0 * SUM(out_strength) / SUM(SUM(out_strength)) OVER (), 1) AS pct_of_all_help
FROM user_metrics
GROUP BY role
ORDER BY help_given DESC;

-- 10. Topic popularity by year (share of each year's questions)
SELECT LEFT(tt.period, 4) AS year, t.label AS topic, SUM(tt.n_questions) AS questions
FROM topic_trends tt
JOIN topics t ON t.topic = tt.topic
GROUP BY LEFT(tt.period, 4), t.label
ORDER BY year, questions DESC;

-- 11. Model results at a glance
SELECT 'prediction' AS task, model AS method, ROUND(roc_auc::numeric, 3) AS score, 'ROC-AUC' AS metric
FROM prediction_metrics
UNION ALL
SELECT 'expert finder', method, ROUND("hit@10"::numeric, 3), 'Hit@10'
FROM recommender_eval
ORDER BY task, score DESC;
