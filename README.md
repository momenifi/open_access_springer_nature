# open_access_springer_nature
This repository contains three files:
 1. data.csv (data.zip): data used for the analyses in the paper 
 2. correlation_cv_feature_importance.py: the source code for the ML analyses.
 3. publications_country_distribution.csv: Distribution of publications based on country of the corresponding author





Data dictionary for data.csv:
CORR_PERSON_ID: author identifier of the corresponding author

GENDER: for male one and female 0, null value for unknown gender status

FIRSTPUB: First publication of the corresponding author

COUNT_CA_PUBLISH_HYB: the number of CA publications in hybrid journals (source Unpaywall)

COUNT_HYBRID_OA_PUBLISH: the number of OA publications in hybrid journals (source Unpaywall)

COUNT_GOLD_PUBLISH: the number of gold OA publications (source Unpaywall)

COUNT_ALL: the number of whole publications

COUNT_OA: number of OA publications (source Unpaywall)

COUNT_CA: the number of CA publications (source Unpaywall)

COUNT_OA_CITING: the number of cited papers with OA status in this publication

COUNT_CA_CITING: the number of cited papers with CA status in this publication

COUNT_UNKNOWN_CITING: the number of cited papers without access status

CNT_INTER_COLLAB: Number of international collaborators in this publication

COAUTHOR_COUNT_ITM: Number of collaborators in this publication

SOURCE_ID: Journal id in Scopus

PUBYEAR: the year of publish

OPEN_ACCESS: Access status of the journal (source springer nature)

COUNTRY_CODE: code of the corresponding author's country

INCOME_GROUP: income level of corresponding author's country

UNPW_IS_OA: if true is OA otherwise CA (Unpaywall)

UNPW_OA_STATUS: five possible access statuses (gold, hybrid, green, bronze, closed )(source Unpaywall)

FIELD: scientific field of publication (the same as the journal's field)

APC_USD: APC of the journal in US dollars

JOURNAL_RANK: journal rank of the journal based on its h-index

AGREEMENT: if transformative agreement with Springer nature is available for the country of the corresponding author. For Germany just for authors afiliated with Max Planck Institues

WAIVER_ELIGIBLE: 1 if the corresponding author is eligible for the APC waiver otherwise 0

DISCOUNT_ELIGIBLE: 1 if the corresponding author is eligible for the APC discount otherwise 0

AVG_GDP_PER_CAPITA: average countries' GDP per capita for years 2017 and 2018
