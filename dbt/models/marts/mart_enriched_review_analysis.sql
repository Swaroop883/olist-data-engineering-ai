
WITH review_analysis AS (

    SELECT *
    FROM {{ ref('mart_review_analysis') }}

),

ai_enrichment AS (

    SELECT
        review_id,
        order_id,
        sentiment AS ai_sentiment,
        topic AS ai_topic,
        summary AS ai_summary,
        issue_category AS ai_issue_category,
        language AS review_language,
        enriched_at

    FROM olist_db.ai.enriched_order_reviews

)

SELECT
    r.review_id,
    r.order_id,
    r.customer_id,
    r.customer_unique_id,

    r.review_score,
    r.review_comment_title,
    r.review_comment_message,

    r.review_creation_date,
    r.review_answer_timestamp,

    r.order_status,
    r.order_purchase_timestamp,

    r.customer_city,
    r.customer_state,

    -- Sentiment derived from the original numeric score
    r.review_sentiment_category,

    -- AI-generated enrichment fields
    ai.ai_sentiment,
    ai.ai_topic,
    ai.ai_summary,
    ai.ai_issue_category,
    ai.review_language,
    ai.enriched_at AS ai_enriched_at

FROM review_analysis AS r

INNER JOIN ai_enrichment AS ai
    ON r.review_id = ai.review_id
    AND r.order_id = ai.order_id
