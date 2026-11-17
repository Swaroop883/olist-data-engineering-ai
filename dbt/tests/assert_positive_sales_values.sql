-- Singular data quality test: Ensure no negative price, freight_value, or total_item_value in mart_sales
select
    order_id,
    order_item_id,
    price,
    freight_value,
    total_item_value
from {{ ref('mart_sales') }}
where price < 0
   or freight_value < 0
   or total_item_value < 0
