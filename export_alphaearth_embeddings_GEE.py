import ee

ee.Initialize()

# CONFIG
REGION_NAME = 'Minnesota'  # US state name (e.g., 'Iowa', 'Illinois', 'Nebraska')
YEAR = 2024
N_SAMPLES_PER_CLASS = 1000
DRIVE_FOLDER = f'AlphaEarth_{REGION_NAME}_{YEAR}'
EXPORT_DESC = f'{REGION_NAME.lower()}_alphaearth_embeddings_CDL_{YEAR}'
EXPORT_PREFIX = f'{REGION_NAME.lower()}_alphaearth_{YEAR}'

# CDL class codes
CORN_CLASS = 1
SOY_CLASS  = 5

# REGION GEOMETRY
states = ee.FeatureCollection('TIGER/2018/States')
region = states.filter(ee.Filter.eq('NAME', REGION_NAME)).geometry()

# USDA CDL 2024
cdl = (
    ee.ImageCollection('USDA/NASS/CDL')
    .filterDate(f'{YEAR}-01-01', f'{YEAR}-12-31')
    .first()
    .select('cropland')
)

# Mask NoData (CDL uses 0 for background / nodata)
cdl = cdl.updateMask(cdl.neq(0))

# CLASS MASKS
corn_mask = cdl.eq(CORN_CLASS)
soy_mask  = cdl.eq(SOY_CLASS)
rest_mask = cdl.neq(CORN_CLASS).And(cdl.neq(SOY_CLASS))

# Create labeled image:
# 0 = corn, 1 = soy, 2 = rest
# Start with invalid label and only assign where CDL is valid
label_img = (
    ee.Image(-1)  # start with invalid label
    .where(cdl.eq(CORN_CLASS), 0)
    .where(cdl.eq(SOY_CLASS), 1)
    .where(cdl.neq(CORN_CLASS).And(cdl.neq(SOY_CLASS)), 2)
    .updateMask(cdl.mask())  # preserve CDL valid pixels only
    .rename('label')
)

# ALPHA EARTH EMBEDDINGS
embeddings = (
    ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
    .filterDate(f'{YEAR}-01-01', f'{YEAR+1}-01-01')
    .mosaic()
)

# STACK FEATURES + LABEL
stack = embeddings.addBands(label_img)

# STRATIFIED SAMPLING
samples = stack.stratifiedSample(
    numPoints=N_SAMPLES_PER_CLASS,
    classBand='label',
    region=region,
    scale=10,
    classValues=[0, 1, 2],
    classPoints=[N_SAMPLES_PER_CLASS]*3,
    geometries=True,
    seed=42,
    tileScale=4
)

# Add human-readable class name
def add_class_name(f):
    label = ee.Number(f.get('label'))
    return f.set('class_name',
        ee.Algorithms.If(label.eq(0), 'corn',
        ee.Algorithms.If(label.eq(1), 'soy', 'rest'))
    )

samples = samples.map(add_class_name)

# EXPORT
task = ee.batch.Export.table.toDrive(
    collection=samples,
    description=EXPORT_DESC,
    folder=DRIVE_FOLDER,
    fileNamePrefix=EXPORT_PREFIX,
    fileFormat='CSV'
)

task.start()

print('Export started:', EXPORT_DESC)
