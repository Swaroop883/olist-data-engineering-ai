with orders as (

    select *
    from {{ ref('core_orders') }}

),

order_summary as (

    select
        customer_unique_id,

        count(distinct order_id) as total_orders,

        min(order_purchase_timestamp) as first_order_date,

        max(order_purchase_timestamp) as last_order_date

    from orders

    group by customer_unique_id

)

select *
from order_summary