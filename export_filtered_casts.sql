-- Create the filtered dataset
CREATE TABLE filtered_casts AS
WITH cast_reactions AS (
    -- Count reactions per cast
    -- TargetCastId format is "fid:hash", so we need to extract the hash part
    SELECT 
        SPLIT_PART(TargetCastId, ':', 2) as target_hash,
        COUNT(*) as reaction_count
    FROM 'data/farcaster_reactions.parquet'
    WHERE TargetCastId IS NOT NULL AND TargetCastId != ''
    GROUP BY SPLIT_PART(TargetCastId, ':', 2)
),
cast_replies AS (
    -- Count replies per cast
    -- ParentCastId format is "fid:hash", so we need to extract the hash part
    SELECT 
        SPLIT_PART(ParentCastId, ':', 2) as parent_hash,
        COUNT(*) as reply_count
    FROM 'data/casts.parquet'
    WHERE ParentCastId IS NOT NULL AND ParentCastId != ''
    GROUP BY SPLIT_PART(ParentCastId, ':', 2)
),
recent_casts AS (
    -- Get casts from last 90 days
    SELECT *
    FROM 'data/casts.parquet'
    WHERE CAST(Timestamp AS BIGINT) >= 133315200
)
SELECT 
    c.*,
    COALESCE(cr.reaction_count, 0) as reaction_count,
    COALESCE(cp.reply_count, 0) as reply_count
FROM recent_casts c
LEFT JOIN cast_reactions cr ON c.Hash = cr.target_hash
LEFT JOIN cast_replies cp ON c.Hash = cp.parent_hash
WHERE 
    -- Looser engagement requirements
    COALESCE(cr.reaction_count, 0) >= 10 AND COALESCE(cr.reaction_count, 0) <= 10000
    AND COALESCE(cp.reply_count, 0) >= 5 AND COALESCE(cp.reply_count, 0) <= 5000
    -- Filter out spam/low-quality content
    AND c.Text IS NOT NULL 
    AND LENGTH(c.Text) >= 20
    AND NOT LOWER(c.Text) LIKE '%airdrop%'
    AND NOT LOWER(c.Text) LIKE '%token%'
    AND NOT LOWER(c.Text) LIKE '%coin%'
    -- Exclude URL-only posts (empty text with embeds)
    AND NOT (c.Text = '' AND c.Embeds IS NOT NULL AND c.Embeds != '');

-- Export to parquet
COPY filtered_casts TO 'data/filtered_casts.parquet' (FORMAT PARQUET);

-- Show summary statistics
SELECT 
    COUNT(*) as total_casts,
    MIN(reaction_count) as min_reactions,
    MAX(reaction_count) as max_reactions,
    AVG(reaction_count) as avg_reactions,
    MIN(reply_count) as min_replies,
    MAX(reply_count) as max_replies,
    AVG(reply_count) as avg_replies
FROM filtered_casts;
