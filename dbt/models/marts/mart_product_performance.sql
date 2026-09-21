with sales as (

    select *
    from {{ ref('mart_sales') }}

),

product_summary as (

    select
        product_id,
        product_category_name,

        count(distinct order_id) as total_orders,
        count(*) as total_items_sold,

        sum(price) as total_product_revenue,
        sum(freight_value) as total_freight_value,
        sum(total_item_value) as total_sales_value,

        avg(price) as average_item_price

    from sales

    group by
        product_id,
        product_category_name

)

select *
from product_summary