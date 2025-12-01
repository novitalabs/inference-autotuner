#!/usr/bin/env node
/**
 * Split agentlog.md by date markers into separate files
 * Usage: node scripts/split_agentlog.js
 */

const fs = require('fs').promises;
const path = require('path');

// Configuration
const INPUT_FILE = 'agentlog.md';
const OUTPUT_DIR = 'agentlog';
const DEFAULT_DATE = { year: '2025', month: '10', day: '21' }; // First commit date
const DATE_PATTERN = /^## (\d{4})\/(\d{2})\/(\d{2})$/;

/**
 * Main function to split agentlog
 */
async function splitAgentlog() {
  console.log('📖 Reading agentlog.md...');

  // Read input file
  const content = await fs.readFile(INPUT_FILE, 'utf-8');
  const lines = content.split('\n');

  console.log(`✅ Loaded ${lines.length} lines`);
  console.log('🔍 Parsing dates and grouping content...');

  // Parse and group content by date
  const dateContents = new Map();
  let currentDate = `${DEFAULT_DATE.year}/${DEFAULT_DATE.month}/${DEFAULT_DATE.day}`;
  let currentLines = [];

  for (const line of lines) {
    const match = line.match(DATE_PATTERN);

    if (match) {
      // Save previous date's content
      if (currentLines.length > 0) {
        const existing = dateContents.get(currentDate) || [];
        existing.push(currentLines.join('\n'));
        dateContents.set(currentDate, existing);
      }

      // Start new date section
      const [_, year, month, day] = match;
      currentDate = `${year}/${month}/${day}`;
      currentLines = [line]; // Include the date marker line
    } else {
      currentLines.push(line);
    }
  }

  // Don't forget the last section
  if (currentLines.length > 0) {
    const existing = dateContents.get(currentDate) || [];
    existing.push(currentLines.join('\n'));
    dateContents.set(currentDate, existing);
  }

  console.log(`📅 Found ${dateContents.size} unique dates`);
  console.log('📁 Creating output directories and files...');

  // Create output directory structure and write files
  let filesCreated = 0;

  for (const [date, contentParts] of dateContents) {
    const [year, month, day] = date.split('/');

    // Create year directory
    const yearDir = path.join(OUTPUT_DIR, year);
    await fs.mkdir(yearDir, { recursive: true });

    // Create filename with leading zeros (mmdd.md)
    const filename = `${month}${day}.md`;
    const filepath = path.join(yearDir, filename);

    // Merge content parts with separator if multiple entries for same date
    const mergedContent = contentParts.join('\n\n---\n\n');

    // Write file
    await fs.writeFile(filepath, mergedContent, 'utf-8');
    filesCreated++;

    console.log(`  ✓ ${filepath}`);
  }

  console.log('');
  console.log('✨ Done!');
  console.log(`📊 Summary:`);
  console.log(`   - Dates processed: ${dateContents.size}`);
  console.log(`   - Files created: ${filesCreated}`);
  console.log(`   - Output directory: ${OUTPUT_DIR}/`);
}

// Run the script
splitAgentlog().catch(error => {
  console.error('❌ Error:', error.message);
  process.exit(1);
});
