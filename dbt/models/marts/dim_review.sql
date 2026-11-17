with reviews as (

    select *
    from {{ ref('mart_review_analysis') }}

),

enriched as (

    select *
    from {{ ref('mart_enriched_review_analysis') }}

),

dedup_reviews as (

    select
        r.review_id,
        r.order_id,
        r.review_score,
        r.review_comment_title,
        r.review_comment_message,
        r.review_sentiment_category,
        e.ai_sentiment,
        e.ai_topic,
        e.ai_summary,
        e.ai_issue_category,
        e.review_language,
        r.review_creation_date,
        r.review_answer_timestamp,
        row_number() over (
            partition by r.review_id
            order by r.review_answer_timestamp desc nulls last, r.order_id
        ) as rn
    from reviews r
    left join enriched e
        on r.review_id = e.review_id
        and r.order_id = e.order_id
    where r.review_id is not null

)

select
    review_id,
    order_id,
    review_score,
    review_comment_title,
    review_comment_message,
    review_sentiment_category,
    ai_sentiment,
    ai_topic,
    ai_summary,
    ai_issue_category,
    review_language,
    review_creation_date,
    review_answer_timestamp
from dedup_reviews
where rn = 1
