const si = require('systeminformation');
const fs = require('node:fs');

si.cpuTemperature().then(data => {
    var content = String(data.main);
    fs.writeFile('./cpu_temp.res', content, err => { if (err) console.error(err); });
}).catch(err => { console.error("Ошибка:", err); });