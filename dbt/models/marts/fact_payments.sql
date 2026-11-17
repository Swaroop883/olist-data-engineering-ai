with payments as (

    select *
    from {{ ref('core_order_payments') }}

)

select
    order_id,
    payment_sequential,
    customer_id,
    customer_unique_id,
    cast(
        strftime(cast(order_purchase_timestamp as date), '%Y%m%d')
        as integer
    ) as date_key,
    order_purchase_timestamp,
    order_status,
    customer_city,
    customer_state,
    payment_type,
    payment_installments,
    payment_value
from payments
