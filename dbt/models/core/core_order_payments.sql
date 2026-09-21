with payments as (

    select *
    from {{ ref('stg_order_payments') }}

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
        p.order_id,
        p.payment_sequential,
        p.payment_type,
        p.payment_installments,
        p.payment_value,

        o.customer_id,
        o.customer_unique_id,
        o.order_status,
        o.order_purchase_timestamp,
        o.customer_city,
        o.customer_state

    from payments p

    left join orders o
        on p.order_id = o.order_id

)

select *
from joined