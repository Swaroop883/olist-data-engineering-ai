with order_items as (

    select *
    from {{ ref('core_order_items') }}

),

orders as (

    select
        order_id,
        customer_id,
        customer_unique_id,
        order_status,
        order_purchase_timestamp,
        customer_state
    from {{ ref('core_orders') }}

),

sales as (

    select
        oi.order_id,
        oi.order_item_id,
        oi.product_id,
        oi.seller_id,

        oi.price,
        oi.freight_value,

        oi.price + oi.freight_value as total_item_value,

        oi.product_category_name,
        oi.seller_state,

        o.customer_id,
        o.customer_unique_id,
        o.order_status,
        o.order_purchase_timestamp,
        o.customer_state

    from order_items oi

    left join orders o
        on oi.order_id = o.order_id

)

select *
from sales