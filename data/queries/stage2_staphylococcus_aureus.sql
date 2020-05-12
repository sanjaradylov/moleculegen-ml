SELECT
  DISTINCT canonical_smiles,
  activities.standard_value
FROM compound_structures
  JOIN activities
    ON compound_structures.molregno = activities.molregno
WHERE
  activities.assay_id IN (

    SELECT assay_id
    FROM assays
    WHERE
      tid IN (

        SELECT tid
        FROM target_dictionary
        WHERE
          pref_name = 'Staphylococcus aureus'
      )

  )

  AND activities.standard_type = 'MIC'
  AND activities.standard_relation = '='
  AND activities.standard_units = 'nM';