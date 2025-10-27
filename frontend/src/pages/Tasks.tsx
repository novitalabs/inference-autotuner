export default function Tasks() {
  return (
    <div className="px-4 py-6 sm:px-0">
      <div className="border-4 border-dashed border-gray-200 rounded-lg p-12">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Tasks</h2>
        <p className="text-gray-600">
          Task management page. This is a placeholder.
        </p>
        <p className="text-gray-600 mt-2">
          Future features:
        </p>
        <ul className="list-disc list-inside text-gray-600 mt-2 ml-4">
          <li>List all tuning tasks</li>
          <li>Create new tasks</li>
          <li>View task details</li>
          <li>Start/stop tasks</li>
          <li>Task status monitoring</li>
        </ul>
      </div>
    </div>
  );
}
