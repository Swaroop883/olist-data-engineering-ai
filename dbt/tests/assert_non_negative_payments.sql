-- Singular data quality test: Ensure no negative payment values exist
select
    order_id,
    payment_sequential,
    payment_value
from {{ ref('core_order_payments') }}
where payment_value < 0
