-- Analysis queries for pgAdmin (Tools -> Query Tool). Run each query on its own:
-- select it and press F5.
--
-- Every community has its own schema: datascience and ai. Queries 1-10 work on one
-- community: run   SET search_path TO datascience;   (or ai) first. The dashboard's
-- Database page does this for you. Queries 11-14 compare both communities and name
-- the schemas explicitly.

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

-- 3. Most used tags and how often their questions go unanswered
SELECT tag,
       COUNT(*)                                                          AS questions,
       ROUND(100.0 * AVG(CASE WHEN q.is_answered THEN 0 ELSE 1 END), 1) AS pct_unanswered
FROM questions q,
     unnest(string_to_array(q.tags, '|')) AS tag
GROUP BY tag
HAVING COUNT(*) >= 20
ORDER BY questions DESC
LIMIT 20;

-- 4. Median hours until the first answer, per quarter
SELECT date_trunc('quarter', q.created_at)::date AS quarter,
       COUNT(*) AS answered_questions,
       ROUND((PERCENTILE_CONT(0.5) WITHIN GROUP (
           ORDER BY EXTRACT(EPOCH FROM fa.first_answer - q.created_at) / 3600))::numeric, 1)
           AS median_hours_to_first_answer
FROM questions q
JOIN (SELECT question_id, MIN(created_at) AS first_answer
      FROM answers GROUP BY question_id) fa ON fa.question_id = q.post_id
GROUP BY quarter
ORDER BY quarter;

-- 5. Learners who first asked, and later became helpers
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
ORDER BY days_to_become_helper
LIMIT 50;

-- 6. Strongest helper -> learner links in the knowledge network
SELECT s.user_name AS helper,
       t.user_name AS learner,
       e.weight    AS interactions
FROM help_edges e
JOIN users s ON s.user_id = e.source_user
LEFT JOIN users t ON t.user_id = e.target_user
ORDER BY e.weight DESC
LIMIT 15;

-- 7. Hardest topics: unanswered rate, waiting time and confusion
SELECT label, category, n_questions,
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

-- 9. Network roles: how many people play each role, and how much help they give
SELECT role,
       COUNT(*)                                                            AS people,
       SUM(out_strength)                                                   AS help_given,
       ROUND(100.0 * SUM(out_strength) / SUM(SUM(out_strength)) OVER (), 1) AS pct_of_all_help
FROM user_metrics
GROUP BY role
ORDER BY help_given DESC;

-- 10. Model results at a glance
SELECT 'answer prediction' AS task, model AS method, ROUND(roc_auc::numeric, 3) AS score, 'ROC-AUC' AS metric
FROM prediction_metrics
UNION ALL
SELECT 'expert finder', method, ROUND("hit@10"::numeric, 3), 'Hit@10'
FROM recommender_eval
ORDER BY task, score DESC;

-- 11. Both communities side by side: size, answered rate, accepted answers
SELECT 'Data Science' AS community, COUNT(*) AS questions,
       ROUND(100.0 * AVG(is_answered::int), 1)                     AS pct_answered,
       ROUND(100.0 * AVG((accepted_answer_id IS NOT NULL)::int), 1) AS pct_accepted
FROM datascience.questions
UNION ALL
SELECT 'Artificial Intelligence', COUNT(*),
       ROUND(100.0 * AVG(is_answered::int), 1),
       ROUND(100.0 * AVG((accepted_answer_id IS NOT NULL)::int), 1)
FROM ai.questions;

-- 12. Topic categories in each community (share of questions)
SELECT 'Data Science' AS community, category,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_of_questions
FROM datascience.question_topics GROUP BY category
UNION ALL
SELECT 'Artificial Intelligence', category,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)
FROM ai.question_topics GROUP BY category
ORDER BY community, pct_of_questions DESC;

-- 13. Knowledge bridges: people active in both communities (same Stack Exchange account)
SELECT d.user_name,
       dm.role AS role_in_data_science, dm.out_strength AS help_in_data_science,
       am.role AS role_in_ai,           am.out_strength AS help_in_ai
FROM datascience.users d
JOIN ai.users a ON a.account_id = d.account_id AND d.account_id > 0
LEFT JOIN datascience.user_metrics dm ON dm.user_id = d.user_id
LEFT JOIN ai.user_metrics am ON am.user_id = a.user_id
ORDER BY COALESCE(dm.out_strength, 0) + COALESCE(am.out_strength, 0) DESC
LIMIT 25;

-- 14. The full comparison table produced by the analysis
SELECT * FROM comparison.overview;
