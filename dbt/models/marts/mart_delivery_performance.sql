with orders as (

    select *
    from {{ ref('core_orders') }}

),

delivery_analysis as (

    select
        order_id,
        customer_id,
        customer_unique_id,

        order_status,

        order_purchase_timestamp,
        order_delivered_carrier_date,
        order_delivered_customer_date,
        order_estimated_delivery_date,

        customer_city,
        customer_state,

        case
            when order_delivered_customer_date is not null
                 and order_estimated_delivery_date is not null
            then date_diff(
                'day',
                order_estimated_delivery_date,
                order_delivered_customer_date
            )
        end as delivery_delay_days

    from orders

)

select *
from delivery_analysis