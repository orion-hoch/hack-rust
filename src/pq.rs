//! Stage 7 implementation: Parquet files through the `parquet` crate's low-level column writer.
//! Plain encoding, no compression, no statistics. Replaceable: callers only use the two
//! functions below and the column structs in `output.rs`.

use std::fs::File;
use std::io::{self, BufWriter};
use std::path::Path;
use std::sync::Arc;

use bytes::Bytes;
use parquet::basic::Compression;
use parquet::data_type::{ByteArray, ByteArrayType, Int32Type, Int64Type};
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::parser::parse_message_type;

use crate::output::{CampaignDaily, NO_CATEGORY, StrColumn, UserDaily};

const USER_DAILY: &str = "message user_daily {
  REQUIRED BYTE_ARRAY user_id (STRING);
  REQUIRED INT32 day (DATE);
  REQUIRED INT64 sessions;
  REQUIRED INT64 events;
  REQUIRED INT64 revenue_cents;
  OPTIONAL BYTE_ARRAY top_category (STRING);
}";

const CAMPAIGN_DAILY: &str = "message campaign_daily {
  REQUIRED BYTE_ARRAY campaign_id (STRING);
  REQUIRED BYTE_ARRAY channel (STRING);
  REQUIRED INT32 day (DATE);
  REQUIRED INT64 sessions;
  REQUIRED INT64 purchasing_sessions;
  REQUIRED INT64 revenue_cents;
  REQUIRED INT64 assisted_sessions;
}";

fn err(e: parquet::errors::ParquetError) -> io::Error {
    io::Error::other(e.to_string())
}

fn props() -> Arc<WriterProperties> {
    Arc::new(
        WriterProperties::builder()
            .set_compression(Compression::UNCOMPRESSED)
            .set_dictionary_enabled(false)
            .set_statistics_enabled(EnabledStatistics::None)
            .build(),
    )
}

fn byte_arrays(col: &StrColumn) -> Vec<ByteArray> {
    let all = Bytes::from(col.data.clone());
    (0..col.len()).map(|i| ByteArray::from(all.slice(col.offsets[i]..col.offsets[i + 1]))).collect()
}

enum Column<'a> {
    Str(Vec<ByteArray>),
    OptStr(Vec<ByteArray>, Vec<i16>),
    I32(&'a [i32]),
    I64(&'a [i64]),
}

fn write(path: &Path, message: &str, columns: Vec<Column>) -> io::Result<()> {
    let schema = Arc::new(parse_message_type(message).map_err(err)?);
    let file = BufWriter::with_capacity(1 << 20, File::create(path)?);
    let mut writer = SerializedFileWriter::new(file, schema, props()).map_err(err)?;
    let mut group = writer.next_row_group().map_err(err)?;
    let mut columns = columns.into_iter();
    while let Some(mut col) = group.next_column().map_err(err)? {
        match columns.next().expect("schema and columns agree") {
            Column::Str(values) => {
                col.typed::<ByteArrayType>().write_batch(&values, None, None).map_err(err)?;
            }
            Column::OptStr(values, defs) => {
                col.typed::<ByteArrayType>().write_batch(&values, Some(&defs), None).map_err(err)?;
            }
            Column::I32(values) => {
                col.typed::<Int32Type>().write_batch(values, None, None).map_err(err)?;
            }
            Column::I64(values) => {
                col.typed::<Int64Type>().write_batch(values, None, None).map_err(err)?;
            }
        }
        col.close().map_err(err)?;
    }
    group.close().map_err(err)?;
    writer.close().map_err(err)?;
    Ok(())
}

pub fn write_user_daily(path: &Path, t: &UserDaily, categories: &[Vec<u8>]) -> io::Result<()> {
    let names: Vec<ByteArray> = categories.iter().map(|c| ByteArray::from(c.clone())).collect();
    let mut defs = Vec::with_capacity(t.len());
    let mut tops = Vec::new();
    for &c in &t.top_category {
        if c == NO_CATEGORY {
            defs.push(0);
        } else {
            defs.push(1);
            tops.push(names[c as usize].clone());
        }
    }
    write(
        path,
        USER_DAILY,
        vec![
            Column::Str(byte_arrays(&t.user_id)),
            Column::I32(&t.day),
            Column::I64(&t.sessions),
            Column::I64(&t.events),
            Column::I64(&t.revenue_cents),
            Column::OptStr(tops, defs),
        ],
    )
}

pub fn write_campaign_daily(path: &Path, t: &CampaignDaily) -> io::Result<()> {
    write(
        path,
        CAMPAIGN_DAILY,
        vec![
            Column::Str(byte_arrays(&t.campaign_id)),
            Column::Str(byte_arrays(&t.channel)),
            Column::I32(&t.day),
            Column::I64(&t.sessions),
            Column::I64(&t.purchasing_sessions),
            Column::I64(&t.revenue_cents),
            Column::I64(&t.assisted_sessions),
        ],
    )
}
