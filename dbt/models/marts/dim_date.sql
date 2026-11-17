with order_dates as (

    select distinct
        cast(order_purchase_timestamp as date) as full_date
    from {{ ref('stg_orders') }}
    where order_purchase_timestamp is not null

    union

    select distinct
        cast(order_delivered_customer_date as date) as full_date
    from {{ ref('stg_orders') }}
    where order_delivered_customer_date is not null

    union

    select distinct
        cast(review_creation_date as date) as full_date
    from {{ ref('stg_order_reviews') }}
    where review_creation_date is not null

)

select
    cast(strftime(full_date, '%Y%m%d') as integer) as date_key,
    full_date,
    extract('year' from full_date) as year,
    extract('quarter' from full_date) as quarter,
    extract('month' from full_date) as month,
    strftime(full_date, '%B') as month_name,
    strftime(full_date, '%Y-%m') as year_month,
    extract('day' from full_date) as day_of_month,
    extract('dow' from full_date) as day_of_week,
    strftime(full_date, '%A') as day_name,
    case
        when extract('dow' from full_date) in (0, 6) then true
        else false
    end as is_weekend
from order_dates
where full_date is not null
order by full_date
