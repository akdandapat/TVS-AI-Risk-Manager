// Node harness to verify every view and all 206 merchant ribbons render without 'undefined' or 'NaN'
const fs = require('fs');
const path = require('path');
const vm = require('vm');

// 1. Stub browser globals
const dummyEl = {
  addEventListener: () => {},
  classList: { add: () => {}, remove: () => {} },
  style: {},
  getBoundingClientRect: () => ({ width: 0, height: 0 }),
  setAttribute: () => {},
  querySelectorAll: () => []
};

global.document = {
  querySelector: () => dummyEl,
  querySelectorAll: () => []
};
global.window = global;
global.innerWidth = 1024;
global.scrollTo = () => {};

// 2. Load data.json
const dataPath = path.join(__dirname, 'web', 'data.json');
const data = JSON.parse(fs.readFileSync(dataPath, 'utf8'));

// 3. Read app.js and strip trailing fetch block
let code = fs.readFileSync(path.join(__dirname, 'web', 'app.js'), 'utf8');
// Strip the fetch block at the end
const fetchIndex = code.indexOf("fetch('data.json')");
if (fetchIndex !== -1) {
  code = code.slice(0, fetchIndex);
}

// 4. Execute app.js in global context
vm.runInThisContext(code);

// Set D to the loaded data
D = data;

// 5. Test all 7 views
const views = ['portfolio', 'watchlist', 'merchant', 'products', 'categories', 'network', 'evidence'];
let errors = [];

console.log('--- Testing 7 Views ---');
for (const v of views) {
  if (typeof R[v] !== 'function') {
    errors.push(`View renderer R['${v}'] is not a function!`);
    continue;
  }
  try {
    const html = R[v]();
    if (typeof html !== 'string' || html.length === 0) {
      errors.push(`View '${v}' produced empty or non-string output!`);
    } else {
      if (html.includes('undefined')) {
        errors.push(`View '${v}' rendered 'undefined' in HTML!`);
      }
      if (html.includes('NaN')) {
        errors.push(`View '${v}' rendered 'NaN' in HTML!`);
      }
      console.log(`✓ View '${v}' passed (${html.length} chars)`);
    }
  } catch (err) {
    errors.push(`View '${v}' threw error: ${err.message}\n${err.stack}`);
  }
}

// 6. Test all 206 merchant ribbons
console.log(`\n--- Testing ${D.merchants.length} Merchant Ribbons ---`);
let ribbonPassCount = 0;
for (let i = 0; i < D.merchants.length; i++) {
  const m = D.merchants[i];
  try {
    const rHtml = ribbon(m);
    if (typeof rHtml !== 'string' || rHtml.length === 0) {
      errors.push(`Merchant ${m.id} ribbon produced empty output!`);
    } else {
      if (rHtml.includes('undefined')) {
        errors.push(`Merchant ${m.id} ribbon rendered 'undefined'!`);
      }
      if (rHtml.includes('NaN')) {
        errors.push(`Merchant ${m.id} ribbon rendered 'NaN'!`);
      }
      ribbonPassCount++;
    }
  } catch (err) {
    errors.push(`Merchant ${m.id} ribbon threw error: ${err.message}`);
  }
}
console.log(`✓ ${ribbonPassCount}/${D.merchants.length} ribbons passed`);

// Also test merchant view for each merchant by setting CUR
console.log(`\n--- Testing merchant() file view for all ${D.merchants.length} merchants ---`);
let merchantFilePassCount = 0;
for (let i = 0; i < D.merchants.length; i++) {
  CUR = D.merchants[i];
  try {
    const mHtml = merchant();
    if (mHtml.includes('undefined')) {
      errors.push(`Merchant ${CUR.id} file rendered 'undefined'!`);
    }
    if (mHtml.includes('NaN')) {
      errors.push(`Merchant ${CUR.id} file rendered 'NaN'!`);
    }
    merchantFilePassCount++;
  } catch (err) {
    errors.push(`Merchant ${CUR.id} file threw error: ${err.message}`);
  }
}
console.log(`✓ ${merchantFilePassCount}/${D.merchants.length} merchant files passed`);

if (errors.length > 0) {
  console.error(`\nFAILED with ${errors.length} errors:`);
  errors.slice(0, 10).forEach(e => console.error('  - ' + e));
  if (errors.length > 10) console.error(`  ... and ${errors.length - 10} more`);
  process.exit(1);
} else {
  console.log('\nAll tests passed successfully with 0 errors, 0 undefined, 0 NaN!');
}
