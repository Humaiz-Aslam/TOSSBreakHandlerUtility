import argparse
import json
import pandas as pd
import trino
import trino.auth
import sys
import os
import urllib3
import math
import warnings
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.simplefilter(action='ignore', category=pd.errors.ParserWarning)


# ---------------- LOGGER ---------------- #
def setup_logger(log_path):
    logger = logging.getLogger("TOSS_HANDLER")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(formatter)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------- CONFIG ---------------- #
def read_appsettings(path="config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_date_from_settings(settings):
    return settings.get("query", {}).get("date")


# ---------------- ORDER MAP ---------------- #
def load_ordermap(path, logger):
    if not path or not os.path.exists(path):
        logger.error("Ordermap CSV not found")
        sys.exit(1)

    logger.info(f"Loading Ordermap CSV: {path}")
    df = pd.read_csv(path, dtype=str)

    if "MarketSession" not in df.columns:
        logger.error("MarketSession column missing in Ordermap CSV")
        sys.exit(1)

    order_map = {}
    for _, r in df.iterrows():
        key = r.iloc[0]
        ms = r["MarketSession"]
        if pd.notna(key) and pd.notna(ms):
            parts = ms.split(":")
            if len(parts) > 1:
                order_map[key] = parts[1]

    logger.info(f"Ordermap loaded | Records: {len(order_map)}")
    return order_map


# ---------------- TRINO CONNECTION ---------------- #
def create_trino_connection(settings, logger):
    auth_cfg = settings.get("authentication", {})
    try:
        return trino.dbapi.connect(
            host=settings.get("host"),
            port=int(settings.get("port")),
            user=auth_cfg.get("username"),
            auth=trino.auth.BasicAuthentication(
                auth_cfg.get("username"),
                auth_cfg.get("password")
            ),
            http_scheme="https",
            verify=settings.get("verify_ssl", True)
        )
    except Exception as e:
        logger.error("Trino connection FAILED")
        logger.error(str(e))
        sys.exit(1)


# ---------------- MEOR PROCESS ---------------- #
def process_meor(settings, df, date_str, logger, chunk_size=10000):
    logger.info("===== MEOR processing STARTED =====")

    mask = (df[3] == "MEOR") & (df.shape[1] > 18)
    ids = df.loc[mask, 16].dropna().unique().tolist()

    logger.info(f"MEOR rows: {mask.sum()} | Unique IDs: {len(ids)}")

    conn = create_trino_connection(settings, logger)
    total_chunks = math.ceil(len(ids) / chunk_size)
    fetched = {}

    for i in range(0, len(ids), chunk_size):
        chunk_no = i // chunk_size + 1
        chunk = ids[i:i + chunk_size]
        logger.info(f"Processing MEOR Chunk {chunk_no}/{total_chunks} | Size: {len(chunk)}")

        q_ids = ",".join(f"'{x}'" for x in chunk)

        query = f"""
        SELECT userdefined_9321, clordid
        FROM datalake.fix.silver_fix
        WHERE orderdate = '{date_str}'
          AND userdefined_9321 IN ({q_ids})
          AND msgtype = 'D'
        """

        try:
            start = time.time()
            cur = conn.cursor()
            cur.execute(query)
            rows = cur.fetchall()
            elapsed = round(time.time() - start, 2)
        except Exception as e:
            logger.error(f"MEOR Trino failure on Chunk {chunk_no}")
            logger.error(str(e))
            sys.exit(1)

        logger.info(f"MEOR Chunk {chunk_no} fetched {len(rows)} rows in {elapsed}s")

        for r in rows:
            fetched[r[0]] = r[1]

    updated_count = 0
    for idx, row in df.loc[mask].iterrows():
        old_val = row[16]
        if old_val in fetched:
            df.at[idx, "OLD_VALUE_COL17"] = old_val
            df.at[idx, 16] = fetched[old_val]
            df.at[idx, "UPDATED"] = "YES"
            updated_count += 1
            logger.info(
                f"MEOR UPDATED | Row {idx} | Column17: {old_val} -> {fetched[old_val]}"
            )

    logger.info(f"MEOR processing COMPLETED | Total Rows Updated: {updated_count}")


# ---------------- MEMR PROCESS ---------------- #
def process_memr(settings, df, date_str, logger, chunk_size=10000):
    logger.info("===== MEMR processing STARTED =====")

    mask = (df[3] == "MEMR") & (df.shape[1] > 18)
    ids = df.loc[mask, 16].dropna().unique().tolist()

    logger.info(f"MEMR rows: {mask.sum()} | Unique IDs: {len(ids)}")

    conn = create_trino_connection(settings, logger)
    total_chunks = math.ceil(len(ids) / chunk_size)
    fetched = {}

    for i in range(0, len(ids), chunk_size):
        chunk_no = i // chunk_size + 1
        chunk = ids[i:i + chunk_size]
        logger.info(f"Processing MEMR Chunk {chunk_no}/{total_chunks} | Size: {len(chunk)}")

        q_ids = ",".join(f"'{x}'" for x in chunk)

        query = f"""
        SELECT userdefined_9321, clordid, origclordid
        FROM datalake.fix.silver_fix
        WHERE orderdate = '{date_str}'
          AND userdefined_9321 IN ({q_ids})
          AND msgtype = 'G'
        """

        try:
            start = time.time()
            cur = conn.cursor()
            cur.execute(query)
            rows = cur.fetchall()
            elapsed = round(time.time() - start, 2)
        except Exception as e:
            logger.error(f"MEMR Trino failure on Chunk {chunk_no}")
            logger.error(str(e))
            sys.exit(1)

        logger.info(f"MEMR Chunk {chunk_no} fetched {len(rows)} rows in {elapsed}s")

        for r in rows:
            fetched[r[0]] = (r[1], r[2])

    updated_count = 0
    for idx, row in df.loc[mask].iterrows():
        old_val = row[16]
        if old_val in fetched:
            clordid, origclordid = fetched[old_val]
            df.at[idx, "OLD_VALUE_COL17"] = old_val
            df.at[idx, 16] = clordid
            df.at[idx, 17] = origclordid
            df.at[idx, "UPDATED"] = "YES"
            updated_count += 1
            logger.info(
                f"MEMR UPDATED | Row {idx} | Column16: {old_val} -> {clordid} | Column17: {origclordid}"
            )

    logger.info(f"MEMR processing COMPLETED | Total Rows Updated: {updated_count}")


# ---------------- MARKET SESSION (UPDATED ONLY) ---------------- #
def apply_market_session(df, order_map, logger):
    logger.info("===== MARKET SESSION UPDATE STARTED =====")

    for idx, row in df.iterrows():
        if row.get("UPDATED") != "YES":
            continue

        if row[3] == "MEOR":
            order_id = row[17]
        elif row[3] == "MEMR":
            order_id = row[16]
        else:
            continue

        if order_id in order_map:
            df.at[idx, 18] = order_map[order_id]

    logger.info("===== MARKET SESSION UPDATE COMPLETED =====")


# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config.json")
    args = parser.parse_args()

    settings = read_appsettings(args.config)
    date_str = get_date_from_settings(settings)

    input_csv = settings.get("InputCSV")
    ordermap_csv = settings.get("Ordermapcsv")

    log_file = os.path.join(os.path.dirname(input_csv), "toss_break_handler.log")
    logger = setup_logger(log_file)

    # Check if the date in config is current
    today_str = datetime.today().strftime("%Y-%m-%d")
    is_current_date = date_str == today_str
    if is_current_date:
        logger.info(f"Date from config is CURRENT DATE: {date_str}")
    else:
        logger.info(f"Date from config is NOT CURRENT DATE: {date_str}")

    with open(input_csv, "r", encoding="utf-8") as f:
        rows = [line.strip().split(",") for line in f.readlines()]

    max_cols = max(len(r) for r in rows)
    rows = [r + [""] * (max_cols - len(r)) for r in rows]
    df = pd.DataFrame(rows, dtype=str)

    with ThreadPoolExecutor(max_workers=3) as executor:
        f_memr = executor.submit(process_memr, settings, df, date_str, logger)
        f_meor = executor.submit(process_meor, settings, df, date_str, logger)
        f_ordermap = executor.submit(load_ordermap, ordermap_csv, logger)

        # Wait for MEMR & MEOR to finish, stop on any exception
        for f in as_completed([f_memr, f_meor]):
            try:
                f.result()
            except Exception:
                sys.exit(1)

        order_map = f_ordermap.result()

    apply_market_session(df, order_map, logger)

    # If date is NOT current, only keep MEMR and MEOR rows
    if not is_current_date:
        df = df[df[3].isin(["MEMR", "MEOR"])]

    output_csv = os.path.join(os.path.dirname(input_csv), "updated_csv.csv")
    df.to_csv(output_csv, index=False, header=False)

    logger.info("===== TOSS Break Handler COMPLETED =====")
