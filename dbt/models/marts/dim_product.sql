with product_categories as (

    select *
    from {{ ref('core_product_categories') }}

)

select
    product_id,
    coalesce(product_category_name, 'unknown') as product_category_name,
    coalesce(
        product_category_name_english,
        product_category_name,
        'unknown'
    ) as product_category_name_english,
    product_name_lenght,
    product_description_lenght,
    product_photos_qty,
    product_weight_g,
    product_length_cm,
    product_height_cm,
    product_width_cm
from product_categories
where product_id is not null
