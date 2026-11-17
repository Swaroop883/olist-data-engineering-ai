with sellers as (

    select *
    from {{ ref('stg_sellers') }}

),

seller_perf as (

    select *
    from {{ ref('mart_seller_performance') }}

)

select
    s.seller_id,
    s.seller_city,
    s.seller_state,
    s.seller_zip_code_prefix,
    coalesce(sp.total_orders, 0) as total_orders,
    coalesce(sp.total_items_sold, 0) as total_items_sold,
    coalesce(sp.total_product_revenue, 0.0) as total_product_revenue,
    coalesce(sp.total_freight_value, 0.0) as total_freight_value,
    coalesce(sp.total_sales_value, 0.0) as total_sales_value,
    coalesce(sp.average_item_price, 0.0) as average_item_price
from sellers s
left join seller_perf sp
    on s.seller_id = sp.seller_id
where s.seller_id is not null
