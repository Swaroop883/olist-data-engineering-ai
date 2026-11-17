with sales as (

    select *
    from {{ ref('mart_sales') }}

),

products as (

    select
        product_id,
        product_category_name_english
    from {{ ref('dim_product') }}

),

delivery as (

    select
        order_id,
        delivery_delay_days
    from {{ ref('mart_delivery_performance') }}

)

select
    s.order_id,
    s.order_item_id,
    s.product_id,
    s.seller_id,
    s.customer_id,
    s.customer_unique_id,
    cast(
        strftime(cast(s.order_purchase_timestamp as date), '%Y%m%d')
        as integer
    ) as date_key,
    s.order_purchase_timestamp,
    s.order_status,
    s.customer_state,
    s.seller_state,
    s.product_category_name,
    coalesce(p.product_category_name_english, s.product_category_name, 'unknown') as product_category_name_english,
    s.price,
    s.freight_value,
    s.total_item_value,
    d.delivery_delay_days
from sales s
left join products p
    on s.product_id = p.product_id
left join delivery d
    on s.order_id = d.order_id
