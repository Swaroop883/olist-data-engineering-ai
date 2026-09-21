with reviews as (

    select *
    from {{ ref('stg_order_reviews') }}

),

orders as (

    select
        order_id,
        customer_id,
        customer_unique_id,
        order_status,
        order_purchase_timestamp,
        customer_city,
        customer_state
    from {{ ref('core_orders') }}

),

joined as (

    select
        r.review_id,
        r.order_id,
        r.review_score,
        r.review_comment_title,
        r.review_comment_message,
        r.review_creation_date,
        r.review_answer_timestamp,

        o.customer_id,
        o.customer_unique_id,
        o.order_status,
        o.order_purchase_timestamp,
        o.customer_city,
        o.customer_state

    from reviews r

    left join orders o
        on r.order_id = o.order_id

)

select *
from joined