with sales as (

    select *
    from {{ ref('mart_sales') }}

)

select
    grouping_id(
        strftime(order_purchase_timestamp, '%Y-%m'),
        customer_state,
        product_category_name
    ) as cuboid_id,
    coalesce(strftime(order_purchase_timestamp, '%Y-%m'), 'ALL_MONTHS') as order_year_month,
    coalesce(customer_state, 'ALL_STATES') as customer_state,
    coalesce(product_category_name, 'ALL_CATEGORIES') as product_category_name,
    count(distinct order_id) as total_orders,
    count(*) as total_items_sold,
    round(sum(price), 2) as total_product_revenue,
    round(sum(freight_value), 2) as total_freight_value,
    round(sum(total_item_value), 2) as total_sales_value,
    round(avg(price), 2) as average_item_price
from sales
group by cube (
    strftime(order_purchase_timestamp, '%Y-%m'),
    customer_state,
    product_category_name
)
