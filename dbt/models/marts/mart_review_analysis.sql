with reviews as (

    select *
    from {{ ref('core_order_reviews') }}

),

review_analysis as (

    select
        review_id,
        order_id,

        customer_id,
        customer_unique_id,

        review_score,
        review_comment_title,
        review_comment_message,

        review_creation_date,
        review_answer_timestamp,

        order_status,
        order_purchase_timestamp,

        customer_city,
        customer_state,

        case
            when review_score >= 4 then 'positive'
            when review_score = 3 then 'neutral'
            when review_score <= 2 then 'negative'
        end as review_sentiment_category

    from reviews

)

select *
from review_analysis