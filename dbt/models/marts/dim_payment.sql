with payment_summary as (

    select *
    from {{ ref('mart_payment_analysis') }}

)

select
    payment_type,
    case payment_type
        when 'credit_card' then 'Credit Card'
        when 'boleto' then 'Boleto Bancario (Cash/Bank Slip)'
        when 'voucher' then 'Gift / Discount Voucher'
        when 'debit_card' then 'Debit Card'
        else 'Other / Not Defined'
    end as payment_type_description,
    case
        when payment_type = 'credit_card' then true
        else false
    end as is_installment_capable,
    total_orders,
    total_payment_value,
    average_payment_value,
    average_installments
from payment_summary
where payment_type is not null
