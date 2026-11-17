with orders as (

    select *
    from {{ ref('stg_orders') }}

),

customers as (

    select *
    from {{ ref('stg_customers') }}

),

joined as (

    select
        o.order_id,
        o.customer_id,
        c.customer_unique_id,

        o.order_status,

        o.order_purchase_timestamp,
        o.order_approved_at,
        o.order_delivered_carrier_date,
        o.order_delivered_customer_date,
        o.order_estimated_delivery_date,

        c.customer_city,
        c.customer_state,
        c.customer_zip_code_prefix

    from orders o

    left join customers c
        on o.customer_id = c.customer_id

)

select *
from joined