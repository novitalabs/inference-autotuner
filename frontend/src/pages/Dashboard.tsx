export default function Dashboard() {
  return (
    <div className="px-4 py-6 sm:px-0">
      <div className="border-4 border-dashed border-gray-200 rounded-lg p-12">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Dashboard</h2>
        <p className="text-gray-600">
          Welcome to the LLM Inference Autotuner dashboard. This is a placeholder page.
        </p>
        <p className="text-gray-600 mt-2">
          Future features:
        </p>
        <ul className="list-disc list-inside text-gray-600 mt-2 ml-4">
          <li>System status overview</li>
          <li>Recent tasks summary</li>
          <li>Performance metrics charts</li>
          <li>Quick actions</li>
        </ul>
      </div>
    </div>
  );
}
