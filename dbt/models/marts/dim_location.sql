with geo as (

    select *
    from {{ ref('core_geolocation') }}
    where geolocation_zip_code_prefix is not null

),

aggregated as (

    select
        geolocation_zip_code_prefix as zip_code_prefix,
        min(geolocation_city) as city,
        min(geolocation_state) as state,
        avg(geolocation_lat) as avg_lat,
        avg(geolocation_lng) as avg_lng
    from geo
    group by geolocation_zip_code_prefix

)

select *
from aggregated
