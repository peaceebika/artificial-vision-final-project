// Region of Interest
var region = ee.Geometry.Rectangle([-100, 18, -97, 21]);

// Sentinel-2 Surface Reflectance
var dataset = ee.ImageCollection('COPERNICUS/S2_SR')
  .filterBounds(region)
  .filterDate('2023-01-01', '2023-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20));

// Median composite
var image = dataset.median();

// Select RGB + NIR bands
var image4band = image.select(['B4', 'B3', 'B2', 'B8']);

// RGB preview
Map.centerObject(region, 8);
Map.addLayer(
  image4band,
  {bands:['B4','B3','B2'], min:0, max:3000},
  'Sentinel-2 RGB Preview'
);

// Export 4-band GeoTIFF
Export.image.toDrive({
  image: image4band,
  description: 'Sentinel2_Mexico_4Band',
  folder: 'computer_vision',
  fileNamePrefix: 'sentinel2_multiband_4band',
  scale: 10,
  region: region,
  fileFormat: 'GeoTIFF',
  maxPixels: 1e13
});