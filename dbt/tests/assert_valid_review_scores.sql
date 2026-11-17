-- Singular data quality test: Ensure all review scores fall within [1, 5]
select
    review_id,
    order_id,
    review_score
from {{ ref('mart_review_analysis') }}
where review_score < 1
   or review_score > 5
   or review_score is null
