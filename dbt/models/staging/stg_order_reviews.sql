with source as (

    select *
    from {{ source('raw', 'raw_order_reviews') }}

),

renamed as (

    select
        review_id,
        order_id,
        review_score,
        review_comment_title,
        review_comment_message,
        review_creation_date,
        review_answer_timestamp

    from source

)

select *
from renamed