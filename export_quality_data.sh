#!/bin/bash

# Export quality cast data using DuckDB CLI for efficient processing

echo "Loading user quality scores and preparing filters..."

# Create temporary SQL files for the export
cat > /tmp/export_quality_casts.sql << 'EOF'
-- First, create views for quality users and included authors
CREATE TEMPORARY TABLE quality_users AS 
SELECT fid 
FROM 'data/user_quality_scores.parquet' 
WHERE quality_score >= 7;

CREATE TEMPORARY TABLE included_authors AS 
SELECT fid 
FROM 'data/user_quality_scores.parquet' 
WHERE quality_score >= 4;

-- Export quality reactions aggregated by cast
COPY (
    SELECT 
        SPLIT_PART(TargetCastId, ':', 2) as target_hash,
        COUNT(*) as total_reactions,
        COUNT(*) FILTER (WHERE r.Fid IN (SELECT fid FROM quality_users)) as quality_reactions
    FROM 'data/farcaster_reactions.parquet' r
    WHERE TargetCastId IS NOT NULL AND TargetCastId != ''
    GROUP BY SPLIT_PART(TargetCastId, ':', 2)
) TO 'data/quality_reactions_temp.parquet' (FORMAT PARQUET);

-- Export quality replies aggregated by cast
COPY (
    SELECT 
        SPLIT_PART(ParentCastId, ':', 2) as parent_hash,
        COUNT(*) as total_replies,
        COUNT(*) FILTER (WHERE c.Fid IN (SELECT fid FROM quality_users)) as quality_replies
    FROM 'data/casts.parquet' c
    WHERE ParentCastId IS NOT NULL AND ParentCastId != ''
    GROUP BY SPLIT_PART(ParentCastId, ':', 2)
) TO 'data/quality_replies_temp.parquet' (FORMAT PARQUET);

-- Export filtered casts with engagement metrics
COPY (
    WITH quality_reactions AS (
        SELECT * FROM 'data/quality_reactions_temp.parquet'
    ),
    quality_replies AS (
        SELECT * FROM 'data/quality_replies_temp.parquet'
    )
    SELECT 
        c.Hash,
        c.Fid,
        c.Text,
        c.Embeds,
        CAST(c.Timestamp AS BIGINT) as timestamp,
        COALESCE(qr.total_reactions, 0) as total_reactions,
        COALESCE(qr.quality_reactions, 0) as quality_reactions,
        COALESCE(qp.total_replies, 0) as total_replies,
        COALESCE(qp.quality_replies, 0) as quality_replies,
        (COALESCE(qr.quality_reactions, 0) * 2 + COALESCE(qp.quality_replies, 0) * 5) as engagement_score
    FROM 'data/casts.parquet' c
    LEFT JOIN quality_reactions qr ON c.Hash = qr.target_hash
    LEFT JOIN quality_replies qp ON c.Hash = qp.parent_hash
    WHERE 
        c.Fid IN (SELECT fid FROM included_authors)
        AND c.Text IS NOT NULL
        AND LENGTH(c.Text) >= 50
        AND CAST(c.Timestamp AS BIGINT) >= 133315200
        -- Only include top-level posts (not replies)
        AND (c.ParentCastId IS NULL OR c.ParentCastId = '')
        -- Require some quality engagement and at least 1 reply
        AND (COALESCE(qr.quality_reactions, 0) >= 3 OR COALESCE(qp.quality_replies, 0) >= 2)
        AND COALESCE(qp.total_replies, 0) > 0
        -- Filter out low-effort patterns
        AND NOT (LOWER(c.Text) LIKE 'gm%' OR LOWER(c.Text) LIKE 'gn%' 
                OR LOWER(c.Text) LIKE 'good morning%' OR LOWER(c.Text) LIKE 'good night%' 
                OR LOWER(c.Text) LIKE 'happy%')
        AND NOT (LENGTH(c.Text) < 100 AND LOWER(c.Text) LIKE '%claim%')
        -- Filter out crypto spam
        AND NOT (c.Text SIMILAR TO '%\$[A-Z]{2,}%')  -- Filters $BTC, $ETH, etc.
        -- Using multiple LIKE conditions for better case-insensitive matching
        AND NOT (LOWER(c.Text) LIKE '%crypto%' OR LOWER(c.Text) LIKE '%blockchain%' 
                OR LOWER(c.Text) LIKE '%defi%' OR LOWER(c.Text) LIKE '%nft%' 
                OR LOWER(c.Text) LIKE '%airdrop%' OR LOWER(c.Text) LIKE '%mint%' 
                OR LOWER(c.Text) LIKE '%whitelist%' OR LOWER(c.Text) LIKE '%presale%' 
                OR LOWER(c.Text) LIKE '%ico%' OR LOWER(c.Text) LIKE '%token%' 
                OR LOWER(c.Text) LIKE '%coin%' OR LOWER(c.Text) LIKE '%hodl%' 
                OR LOWER(c.Text) LIKE '%moon%' OR LOWER(c.Text) LIKE '%pump%' 
                OR LOWER(c.Text) LIKE '%dump%' OR LOWER(c.Text) LIKE '%wagmi%' 
                OR LOWER(c.Text) LIKE '%gm ser%' OR LOWER(c.Text) LIKE '%lfg%' 
                OR LOWER(c.Text) LIKE '%dyor%' OR LOWER(c.Text) LIKE '%fomo%' 
                OR LOWER(c.Text) LIKE '%fud%' OR LOWER(c.Text) LIKE '%ape%' 
                OR LOWER(c.Text) LIKE '%degen%' OR LOWER(c.Text) LIKE '%yield farm%' 
                OR LOWER(c.Text) LIKE '%liquidity pool%' OR LOWER(c.Text) LIKE '%staking%' 
                OR LOWER(c.Text) LIKE '%rug%' OR LOWER(c.Text) LIKE '%scam%' 
                OR LOWER(c.Text) LIKE '%moxie%' OR LOWER(c.Text) LIKE '%win%' 
                OR LOWER(c.Text) LIKE '%farcaster%' OR LOWER(c.Text) LIKE '%neynar%' 
                OR LOWER(c.Text) LIKE '%warpcast%' OR LOWER(c.Text) LIKE '%farcon%' 
                OR LOWER(c.Text) LIKE '%tribe%' OR LOWER(c.Text) LIKE '%ctg%' 
                OR LOWER(c.Text) LIKE '%farhack%' OR LOWER(c.Text) LIKE '%rewards%')
        -- Filter out eggs/cock/hen spam
        AND NOT (LOWER(c.Text) LIKE '%egg%' OR LOWER(c.Text) LIKE '%eggs%' 
                OR LOWER(c.Text) LIKE '%cock%' OR LOWER(c.Text) LIKE '%hen%' 
                OR LOWER(c.Text) LIKE '%hens%' OR LOWER(c.Text) LIKE '%chicken%' 
                OR LOWER(c.Text) LIKE '%rooster%' OR LOWER(c.Text) LIKE '%cluck%' 
                OR LOWER(c.Text) LIKE '%lay egg%' OR LOWER(c.Text) LIKE '%laid egg%')
        -- Filter out gambling/spin spam
        AND NOT (LOWER(c.Text) LIKE '%spin%' OR LOWER(c.Text) LIKE '%warpslot%' 
                OR LOWER(c.Text) LIKE '%slot%' OR LOWER(c.Text) LIKE '%casino%' 
                OR LOWER(c.Text) LIKE '%lottery%' OR LOWER(c.Text) LIKE '%jackpot%' 
                OR LOWER(c.Text) LIKE '%won%' OR LOWER(c.Text) LIKE '%win big%' 
                OR LOWER(c.Text) LIKE '%free spin%' OR LOWER(c.Text) LIKE '%daily spin%' 
                OR LOWER(c.Text) LIKE '%spin to win%' OR LOWER(c.Text) LIKE '%lucky%' 
                OR LOWER(c.Text) LIKE '%fortune%')
        -- Filter out follow/recast spam
        AND NOT (LOWER(c.Text) LIKE '%follow me%' OR LOWER(c.Text) LIKE '%follow back%' 
                OR LOWER(c.Text) LIKE '%recast%' OR LOWER(c.Text) LIKE '%like and recast%' 
                OR LOWER(c.Text) LIKE '%share to win%' OR LOWER(c.Text) LIKE '%tag friends%' 
                OR LOWER(c.Text) LIKE '%follow for follow%' OR LOWER(c.Text) LIKE '%f4f%')
    ORDER BY engagement_score DESC
    LIMIT 200000
) TO 'data/quality_casts_raw.parquet' (FORMAT PARQUET);

-- Script complete
EOF

echo "Exporting quality casts data using DuckDB CLI..."
echo "This will utilize available RAM more efficiently..."

# Run DuckDB with increased memory limit
duckdb -c ".read /tmp/export_quality_casts.sql"

# Clean up
rm -f data/quality_reactions_temp.parquet data/quality_replies_temp.parquet
rm -f /tmp/export_quality_casts.sql

echo "Export complete! Data saved to data/quality_casts_raw.parquet"
echo "Now run: python filter_quality_content.py"