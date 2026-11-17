with orders as (

    select *
    from {{ ref('core_orders') }}

),

customer_dedup as (

    select
        customer_unique_id,
        customer_id as latest_customer_id,
        customer_city,
        customer_state,
        customer_zip_code_prefix,
        row_number() over (
            partition by customer_unique_id
            order by order_purchase_timestamp desc nulls last, customer_id
        ) as rn
    from orders
    where customer_unique_id is not null

),

behavior as (

    select *
    from {{ ref('mart_customer_behavior') }}

)

select
    c.customer_unique_id,
    c.latest_customer_id as customer_id,
    c.customer_city,
    c.customer_state,
    c.customer_zip_code_prefix,
    coalesce(b.total_orders, 0) as total_orders,
    b.first_order_date,
    b.last_order_date,
    case
        when coalesce(b.total_orders, 0) > 1 then true
        else false
    end as is_repeat_customer
from customer_dedup c
left join behavior b
    on c.customer_unique_id = b.customer_unique_id
where c.rn = 1
