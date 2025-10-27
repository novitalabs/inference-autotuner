import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "react-hot-toast";
import Layout from "@/components/Layout";

const queryClient = new QueryClient({
	defaultOptions: {
		queries: {
			refetchOnWindowFocus: false,
			retry: 1
		}
	}
});

function App() {
	return (
		<QueryClientProvider client={queryClient}>
			<Layout />
			<Toaster />
		</QueryClientProvider>
	);
}

export default App;
