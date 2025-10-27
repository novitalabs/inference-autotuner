export default function Experiments() {
  return (
    <div className="px-4 py-6 sm:px-0">
      <div className="border-4 border-dashed border-gray-200 rounded-lg p-12">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Experiments</h2>
        <p className="text-gray-600">
          Experiment results page. This is a placeholder.
        </p>
        <p className="text-gray-600 mt-2">
          Future features:
        </p>
        <ul className="list-disc list-inside text-gray-600 mt-2 ml-4">
          <li>View all experiments</li>
          <li>Filter experiments by task</li>
          <li>Compare experiment results</li>
          <li>Visualize metrics</li>
          <li>Export results</li>
        </ul>
      </div>
    </div>
  );
}
