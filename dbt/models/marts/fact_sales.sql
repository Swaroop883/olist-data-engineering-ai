select *
from {{ ref('fact_order_items') }}
