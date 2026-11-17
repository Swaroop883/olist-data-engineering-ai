with reviews as (

    select *
    from {{ ref('mart_review_analysis') }}

),

enriched as (

    select *
    from {{ ref('mart_enriched_review_analysis') }}

),

delivery as (

    select
        order_id,
        delivery_delay_days
    from {{ ref('mart_delivery_performance') }}

)

select
    r.review_id,
    r.order_id,
    r.customer_id,
    r.customer_unique_id,
    cast(
        strftime(cast(r.review_creation_date as date), '%Y%m%d')
        as integer
    ) as date_key,
    r.review_creation_date,
    r.review_answer_timestamp,
    r.order_status,
    r.order_purchase_timestamp,
    r.customer_city,
    r.customer_state,
    r.review_score,
    r.review_sentiment_category,
    e.ai_sentiment,
    e.ai_topic,
    e.ai_issue_category,
    e.review_language,
    d.delivery_delay_days
from reviews r
left join enriched e
    on r.review_id = e.review_id
    and r.order_id = e.order_id
left join delivery d
    on r.order_id = d.order_id
