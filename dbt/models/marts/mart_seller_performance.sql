with sales as (

    select *
    from {{ ref('mart_sales') }}

),

seller_summary as (

    select
        seller_id,
        seller_state,

        count(distinct order_id) as total_orders,
        count(*) as total_items_sold,

        sum(price) as total_product_revenue,
        sum(freight_value) as total_freight_value,
        sum(total_item_value) as total_sales_value,

        avg(price) as average_item_price

    from sales

    group by
        seller_id,
        seller_state

)

select *
from seller_summary