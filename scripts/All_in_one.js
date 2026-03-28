// =====================================================================
// MODIS + SENTINEL-2 (TOA+SR) + LANDSAT 8&9 — SESHACHALAM FOREST
// S2: TOA (2015-2016, QA60) + SR (2017-2026, SCL) merged for max data
// Day intervals: MODIS~8day, S2~5day, Landsat~8day
// =====================================================================

var SITE_NAME = 'seshachalam_forest';

// ── SESHACHALAM FOREST COORDINATES ───────────────────────────────────
var LONGITUDE = 79.16;
var LATITUDE  = 13.813;

var aoi = ee.Geometry.Polygon([[
  [79.14, 13.800],
  [79.18, 13.800],
  [79.18, 13.826],
  [79.14, 13.826],
  [79.14, 13.800]
]]);

var point = ee.Geometry.Point([LONGITUDE, LATITUDE]);

Map.centerObject(aoi, 13);
Map.addLayer(aoi,   {color: 'green'}, 'Seshachalam Forest AOI');
Map.addLayer(point, {color: 'red'},   'Centre Point (79.16, 13.813)');

// ── END DATE ──────────────────────────────────────────────────────────
var END_DATE = '2026-03-31';

// ── SENSOR START DATES ────────────────────────────────────────────────
var MODIS_TERRA_START = '2000-02-18';
var MODIS_AQUA_START  = '2002-07-04';
var S2_TOA_START      = '2015-06-23';  // TOA available from launch
var S2_SR_START       = '2017-01-01';  // SR reliably available from 2017
var L8_START          = '2013-04-11';
var L9_START          = '2021-10-31';

// =====================================================================
// 1. MODIS TERRA + AQUA | ~8-day merged | 250m | 2000 → 2026
// =====================================================================
function prepMODIS(img) {
  var mask = img.select('SummaryQA').lte(1);
  var ndvi = img.select('NDVI').multiply(0.0001).rename('NDVI');
  return img.addBands(ndvi, null, true)
            .updateMask(mask)
            .copyProperties(img, ['system:time_start']);
}

var terra = ee.ImageCollection('MODIS/061/MOD13Q1')
  .filterDate(MODIS_TERRA_START, END_DATE)
  .filterBounds(aoi)
  .map(prepMODIS);

var aqua = ee.ImageCollection('MODIS/061/MYD13Q1')
  .filterDate(MODIS_AQUA_START, END_DATE)
  .filterBounds(aoi)
  .map(prepMODIS);

var modis = terra.merge(aqua).sort('system:time_start');

var modisTS = modis.map(function(img) {
  var val = img.select('NDVI').reduceRegion({
    reducer  : ee.Reducer.mean(),
    geometry : aoi,
    scale    : 250,
    maxPixels: 1e9
  });
  return ee.Feature(null, {
    'date'  : ee.Date(img.get('system:time_start')).format('YYYY-MM-dd'),
    'NDVI'  : val.get('NDVI'),
    'sensor': 'MODIS'
  });
}).filter(ee.Filter.notNull(['NDVI']));

// =====================================================================
// 2. SENTINEL-2 | ~5-day | 2015 → 2026
//    PART A: TOA / Level-1C (2015-06-23 → 2016-12-31) — QA60 mask
//    PART B: SR  / Level-2A (2017-01-01 → 2026-03-31) — SCL  mask
//    Merged → full continuous record from launch to present
// =====================================================================

// ── PART A: TOA (2015–2016) — QA60 cloud mask ────────────────────────
function maskS2_TOA(img) {
  var qa  = img.select('QA60');
  var cloudBitMask  = 1 << 10;  // opaque clouds
  var cirrusBitMask = 1 << 11;  // cirrus clouds
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
               .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return img.updateMask(mask)
            .divide(10000)       // TOA reflectance scale
            .copyProperties(img, ['system:time_start']);
}

function addNDVI_S2(img) {
  return img.addBands(
    img.select('B8').subtract(img.select('B4'))
       .divide(img.select('B8').add(img.select('B4')))
       .rename('NDVI')
  );
}

var s2_toa = ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
  .filterDate(S2_TOA_START, '2016-12-31')   // TOA only for 2015–2016
  .filterBounds(aoi)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80))
  .map(maskS2_TOA)
  .map(addNDVI_S2)
  .sort('system:time_start');

// ── PART B: SR (2017–2026) — SCL pixel-level mask ────────────────────
function maskS2_SR(img) {
  var scl  = img.select('SCL');
  // Keep: 4=vegetation, 5=bare soil, 6=water, 7=unclassified
  // Drop: 3=cloud shadow, 8=cloud med, 9=cloud high, 10=cirrus, 11=snow
  var mask = scl.eq(4).or(scl.eq(5)).or(scl.eq(6)).or(scl.eq(7));
  return img.updateMask(mask)
            .divide(10000)       // SR reflectance scale
            .copyProperties(img, ['system:time_start']);
}

var s2_sr = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterDate(S2_SR_START, END_DATE)         // SR from 2017 onwards
  .filterBounds(aoi)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80))
  .map(maskS2_SR)
  .map(addNDVI_S2)
  .sort('system:time_start');

// ── MERGE TOA + SR → full 2015–2026 record ───────────────────────────
var s2 = s2_toa.merge(s2_sr).sort('system:time_start');

var s2TS = s2.map(function(img) {
  var val = img.select('NDVI').reduceRegion({
    reducer  : ee.Reducer.mean(),
    geometry : aoi,
    scale    : 10,
    maxPixels: 1e9
  });
  return ee.Feature(null, {
    'date'  : ee.Date(img.get('system:time_start')).format('YYYY-MM-dd'),
    'NDVI'  : val.get('NDVI'),
    'sensor': 'Sentinel-2'
  });
}).filter(ee.Filter.notNull(['NDVI']));

// =====================================================================
// 3. LANDSAT 8 + 9 | ~8-day merged | 30m | 2013 → 2026
// =====================================================================
function maskL(img) {
  var qa   = img.select('QA_PIXEL');
  // Bit 3 = cloud shadow, Bit 4 = cloud
  var mask = qa.bitwiseAnd(1 << 3).eq(0)
               .and(qa.bitwiseAnd(1 << 4).eq(0));
  return img.updateMask(mask)
            .multiply(0.0000275).add(-0.2)  // correct C02 scale factor
            .copyProperties(img, ['system:time_start']);
}

function addNDVI_L(img) {
  return img.addBands(
    img.select('SR_B5').subtract(img.select('SR_B4'))
       .divide(img.select('SR_B5').add(img.select('SR_B4')))
       .rename('NDVI')
  );
}

function prepL(col) {
  return col.filterBounds(aoi)
            .filter(ee.Filter.lt('CLOUD_COVER', 80))
            .map(maskL)
            .map(addNDVI_L);
}

var l8 = prepL(
  ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterDate(L8_START, END_DATE)
);

var l9 = prepL(
  ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
    .filterDate(L9_START, END_DATE)
);

var landsat = l8.merge(l9).sort('system:time_start');

var landsatTS = landsat.map(function(img) {
  var val = img.select('NDVI').reduceRegion({
    reducer  : ee.Reducer.mean(),
    geometry : aoi,
    scale    : 30,
    maxPixels: 1e9
  });
  return ee.Feature(null, {
    'date'  : ee.Date(img.get('system:time_start')).format('YYYY-MM-dd'),
    'NDVI'  : val.get('NDVI'),
    'sensor': 'Landsat'
  });
}).filter(ee.Filter.notNull(['NDVI']));

// =====================================================================
// PRINT SCENE COUNTS
// =====================================================================
print('── Seshachalam Forest — Scene Counts ────────────────');
print('Centre : 79.16°E, 13.813°N | Area: ~12.5 km²');
print('─────────────────────────────────────────────────────');
print('MODIS Terra  (16-day | 2000→2026):', terra.size());
print('MODIS Aqua   (16-day | 2002→2026):', aqua.size());
print('MODIS Merged (~8-day effective)  :', modis.size());
print('─────────────────────────────────────────────────────');
print('S2 TOA  (QA60 | 2015–2016)       :', s2_toa.size());
print('S2 SR   (SCL  | 2017–2026)       :', s2_sr.size());
print('S2 Total (~5-day | 2015→2026)    :', s2.size());
print('─────────────────────────────────────────────────────');
print('Landsat 8    (16-day | 2013→2026):', l8.size());
print('Landsat 9    (16-day | 2021→2026):', l9.size());
print('L8+L9 Merged (~8-day effective)  :', landsat.size());
print('─────────────────────────────────────────────────────');

// =====================================================================
// CHARTS
// =====================================================================
print('── NDVI Charts ──────────────────────────────────────');

print(ui.Chart.feature.byFeature(modisTS, 'date', ['NDVI'])
  .setChartType('LineChart')
  .setOptions({
    title    : 'MODIS NDVI — Seshachalam Forest (~8-day | 2000–2026)',
    hAxis    : {title: 'Date'},
    vAxis    : {title: 'NDVI', viewWindow: {min: 0, max: 1}},
    lineWidth: 1, pointSize: 2, colors: ['#e65100']
  }));

print(ui.Chart.feature.byFeature(s2TS, 'date', ['NDVI'])
  .setChartType('LineChart')
  .setOptions({
    title    : 'Sentinel-2 NDVI — Seshachalam Forest (~5-day | 2015–2026) [TOA+SR]',
    hAxis    : {title: 'Date'},
    vAxis    : {title: 'NDVI', viewWindow: {min: 0, max: 1}},
    lineWidth: 1, pointSize: 2, colors: ['#1565c0']
  }));

print(ui.Chart.feature.byFeature(landsatTS, 'date', ['NDVI'])
  .setChartType('LineChart')
  .setOptions({
    title    : 'Landsat 8+9 NDVI — Seshachalam Forest (~8-day | 2013–2026)',
    hAxis    : {title: 'Date'},
    vAxis    : {title: 'NDVI', viewWindow: {min: 0, max: 1}},
    lineWidth: 1, pointSize: 3, colors: ['#2e7d32']
  }));

// =====================================================================
// 3 EXPORT TASKS → Google Drive
// =====================================================================
print('── Exports (check Tasks tab) ────────────────────────');

Export.table.toDrive({
  collection : ee.FeatureCollection(modisTS),
  description: SITE_NAME + '_MODIS_NDVI_8day_2000_to_Mar2026',
  folder     : 'GEE_exports',
  fileFormat : 'CSV',
  selectors  : ['date', 'NDVI', 'sensor']
});
print('1. seshachalam_MODIS_NDVI_8day_2000_to_Mar2026.csv');

Export.table.toDrive({
  collection : ee.FeatureCollection(s2TS),
  description: SITE_NAME + '_S2_NDVI_5day_2015_to_Mar2026_TOA_SR',
  folder     : 'GEE_exports',
  fileFormat : 'CSV',
  selectors  : ['date', 'NDVI', 'sensor']
});
print('2. seshachalam_S2_NDVI_5day_2015_to_Mar2026_TOA_SR.csv');

Export.table.toDrive({
  collection : ee.FeatureCollection(landsatTS),
  description: SITE_NAME + '_L8L9_NDVI_8day_2013_to_Mar2026',
  folder     : 'GEE_exports',
  fileFormat : 'CSV',
  selectors  : ['date', 'NDVI', 'sensor']
});
print('3. seshachalam_L8L9_NDVI_8day_2013_to_Mar2026.csv');

print('✓ Go to Tasks tab → click Run on all 3 tasks');
