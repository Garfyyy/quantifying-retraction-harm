import polars as pl

from functools import reduce

target_df = pl.read_parquet('./res/paper_c1/quantize_0_0.parquet')

paper_vector = target_df.select(["corpusid", "data", "fieldid", "year"])
year_1_10_col = [f"{i}" for i in range(1, 11)]

paper_vector = paper_vector.with_columns([pl.col("data").list.get(i).alias(f"{i}") for i in range(11)]).drop("data").drop_nulls(subset=["0"])

conditions = [pl.col(i).is_null() for i in year_1_10_col]
filed_match_expr = reduce(lambda a, b: a & b, conditions)

paper_vector = paper_vector.filter(~(filed_match_expr))

paper_field_explode = paper_vector.explode("fieldid")
paper_field_explode = paper_field_explode.with_columns(
    pl.when(pl.col("fieldid") == 1).then(2).otherwise(pl.col("fieldid")).alias("fieldid")
)

# for 'Medicine' field
paper_field_explode_test = paper_field_explode.filter(pl.col("fieldid") == 19).select(year_1_10_col)

paper_field_explode_test.with_columns([(1 - pl.col(i)).alias(i) for i in year_1_10_col])

q1 = paper_field_explode_test.select(year_1_10_col).quantile(0.25).to_numpy().flatten()
q2 = paper_field_explode_test.select(year_1_10_col).quantile(0.5).to_numpy().flatten()
q3 = paper_field_explode_test.select(year_1_10_col).quantile(0.75).to_numpy().flatten()

print(q1, q2, q3)