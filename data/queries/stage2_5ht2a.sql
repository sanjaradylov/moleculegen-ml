/*
  Retrieve canonical SMILES strings and 5-HT2A activities for Stage 2.
*/

WITH target_ids AS (
  SELECT tid
  FROM target_dictionary
  WHERE pref_name IN (
    'Serotonin 2a (5-HT2a) receptor',
    '5-hydroxytryptamine receptor 2A'
  )
), assay_ids AS (
  SELECT assay_id
  FROM assays
  WHERE tid IN target_ids
), activity_data AS (
  SELECT
    DISTINCT molregno,
    standard_value
  FROM activities
  WHERE
    assay_id IN assay_ids
    AND standard_type = 'IC50'
    AND standard_relation = '='
    AND standard_units = 'nM'
)

SELECT
  DISTINCT compound_structures.canonical_smiles,
  activity_data.standard_value
FROM compound_structures
  JOIN activity_data
    ON compound_structures.molregno = activity_data.molregno;