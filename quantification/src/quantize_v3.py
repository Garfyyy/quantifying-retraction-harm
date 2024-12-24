import gc
import argparse

from functools import reduce

import numpy as np
import polars as pl
from tqdm.auto import tqdm


MAX_YEAR = 2023
RANGE_YEAR = 10
T_YEAR = MAX_YEAR - RANGE_YEAR

citations_n_year = pl.read_parquet(f"./citations_10_year_with_v_fl_id_fillnull_new.parquet")

# print(re_citations_ids_df.shape)
selecet_col = [str(i + 1) for i in range(10)]

def process_chunk(chunk: pl.DataFrame) -> list:
    res = []
    for row in tqdm(chunk.iter_rows(named=True), total=chunk.shape[0]):
        c_id = row["corpusid"]
        c_venue = row["venueid"]
        c_year = row["year"]
        c_field = row["fieldid"]
        c_citationcount_sum = row["citationcount"]
        c_vector = np.array([row[str(i)] for i in range(1, 11)])

        conditions = [pl.col("fieldid").list.contains(field) for field in c_field]
        filed_match_expr = reduce(lambda a, b: a | b, conditions)
        
        citations_n_year_d = citations_n_year.filter(pl.col("venueid") == c_venue)
        citations_n_year_d = citations_n_year_d.filter((pl.col("corpusid") != c_id))
        citations_n_year_d = citations_n_year_d.filter(abs(pl.col("year") - c_year) <= 1)
        citations_n_year_d = citations_n_year_d.filter((filed_match_expr))

        d_n = citations_n_year_d.shape[0]
        if d_n == 0:
            continue
        
        d_citationcount_sum = citations_n_year_d["citationcount"].sum()

        citations_n_year_d_res = citations_n_year_d.select(selecet_col).sum().to_numpy()[0]

        c_res = [
                c_citationcount_sum / (d_citationcount_sum / d_n)
                if d_citationcount_sum != 0
                else None
        ] + [
            c / (d / d_n) if d != 0 else None
            for c, d in zip(c_vector, citations_n_year_d_res)
        ]

        res.append({
            "corpusid" : c_id,
            "venueid": c_venue,
            "year": c_year,
            "citationcount": c_citationcount_sum,
            "fieldid": c_field,
            "data": c_res,
        })
        
    return res

if __name__ == "__main__":
    parrser = argparse.ArgumentParser()
    
    parrser.add_argument
    parrser.add_argument("--n_chunk", type=int)
    # parrser.add_argument("--chunk_size", type=int)
    parrser.add_argument("--n_cite", type=int)
    parrser.add_argument("--total_chunk", type=int)
    
    args = parrser.parse_args()

    # re_citations_df = pl.read_parquet(f"../ref_per_process/retractions_citations_{args.n_cite}_n.parquet")
    re_citations_df = pl.read_parquet(f"./ref_per_process/references_{args.n_cite}.parquet")
    re_citations_df = citations_n_year.join(re_citations_df.select("citingcorpusid"), left_on="corpusid", right_on="citingcorpusid", how="semi")
    
    sucess_df = pl.DataFrame()
    if args.n_cite:
        sucess_df = pl.read_parquet(f"./res/paper_c1/paper_c1.parquet")
        for i in range(2, args.n_cite + 1):
            cur_df = pl.read_parquet(f"./res/paper_c{i}/paper_c{i}.parquet")
            sucess_df = pl.concat([sucess_df, cur_df]).unique("corpusid")
        
        sucess_df = sucess_df.join(re_citations_df.select("corpusid"), left_on="corpusid", right_on="corpusid", how="semi")
        re_citations_df = re_citations_df.join(sucess_df.select("corpusid"), left_on="corpusid", right_on="corpusid", how="anti")
        
    chunk_size = re_citations_df.shape[0] // args.total_chunk
    
    chunk = re_citations_df.slice(args.n_chunk * chunk_size, chunk_size)

    if chunk.shape[0] == 0:
        exit(0)
    
    del re_citations_df
    gc.collect()
    
    res = process_chunk(chunk)
    res_df = pl.DataFrame(res)

    if sucess_df.is_empty() or args.n_chunk:
        sucess_df = res_df
    else:
        sucess_df = pl.concat([sucess_df, res_df]).unique("corpusid")
        
    sucess_df.write_parquet(f"./res/quantize_{args.n_cite}_{args.n_chunk}.parquet")