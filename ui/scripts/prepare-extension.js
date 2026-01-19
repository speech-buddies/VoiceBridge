const fs = require('fs');
const path = require('path');

const buildDir = path.join(__dirname, '..', 'build');
const extensionDir = path.join(__dirname, '..', 'extension');

// Ensure extension directory exists
if (!fs.existsSync(extensionDir)) {
  fs.mkdirSync(extensionDir, { recursive: true });
}

// Copy static files
const staticDir = path.join(buildDir, 'static');
const extensionStaticDir = path.join(extensionDir, 'static');

if (fs.existsSync(staticDir)) {
  // Remove old static directory
  if (fs.existsSync(extensionStaticDir)) {
    fs.rmSync(extensionStaticDir, { recursive: true, force: true });
  }
  
  // Copy new static files
  copyRecursiveSync(staticDir, extensionStaticDir);
}

// Create asset-manifest.json in the format expected by content script
const assetManifest = path.join(buildDir, 'asset-manifest.json');
const extensionAssetManifest = path.join(extensionDir, 'asset-manifest.json');

if (fs.existsSync(assetManifest)) {
  const manifest = JSON.parse(fs.readFileSync(assetManifest, 'utf8'));
  
  // Extract JS and CSS files from entrypoints
  const jsFiles = manifest.entrypoints
    .filter(file => file.endsWith('.js'))
    .map(file => file.replace(/^\//, '')); // Remove leading slash
  
  const cssFiles = manifest.entrypoints
    .filter(file => file.endsWith('.css'))
    .map(file => file.replace(/^\//, '')); // Remove leading slash
  
  // Create simplified manifest for extension
  const extensionManifest = {
    js: jsFiles,
    css: cssFiles
  };
  
  fs.writeFileSync(extensionAssetManifest, JSON.stringify(extensionManifest, null, 2));
}

// Copy extension-specific files (always copy to ensure latest version)
const extensionFiles = [
  'manifest.json',
  'background.js',
  'content.js',
  'content.css',
  'popup.html',
  'popup.js'
];

extensionFiles.forEach(file => {
  const srcFile = path.join(__dirname, '..', file);
  const destFile = path.join(extensionDir, file);
  
  // Copy if source exists
  if (fs.existsSync(srcFile)) {
    fs.copyFileSync(srcFile, destFile);
  }
});

// Ensure icons directory exists
const iconsDir = path.join(extensionDir, 'icons');
if (!fs.existsSync(iconsDir)) {
  fs.mkdirSync(iconsDir, { recursive: true });
}

console.log('Extension files prepared in:', extensionDir);

function copyRecursiveSync(src, dest) {
  const exists = fs.existsSync(src);
  const stats = exists && fs.statSync(src);
  const isDirectory = exists && stats.isDirectory();
  
  if (isDirectory) {
    if (!fs.existsSync(dest)) {
      fs.mkdirSync(dest, { recursive: true });
    }
    fs.readdirSync(src).forEach(childItemName => {
      copyRecursiveSync(
        path.join(src, childItemName),
        path.join(dest, childItemName)
      );
    });
  } else {
    fs.copyFileSync(src, dest);
  }
}
