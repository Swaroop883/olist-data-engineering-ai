with payments as (

    select *
    from {{ ref('core_order_payments') }}

),

payment_summary as (

    select
        payment_type,

        count(distinct order_id) as total_orders,

        sum(payment_value) as total_payment_value,

        avg(payment_value) as average_payment_value,

        avg(payment_installments) as average_installments

    from payments

    group by payment_type

)

select *
from payment_summary