with reviews as (

    select *
    from {{ ref('mart_review_analysis') }}

),

customer_summary as (

    select
        customer_unique_id,

        count(distinct review_id) as total_reviews,

        avg(review_score) as average_review_score,

        sum(
            case
                when review_sentiment_category = 'positive' then 1
                else 0
            end
        ) as positive_reviews,

        sum(
            case
                when review_sentiment_category = 'neutral' then 1
                else 0
            end
        ) as neutral_reviews,

        sum(
            case
                when review_sentiment_category = 'negative' then 1
                else 0
            end
        ) as negative_reviews

    from reviews

    group by customer_unique_id

)

select *
from customer_summary