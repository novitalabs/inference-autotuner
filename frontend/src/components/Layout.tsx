import { useState, useEffect } from "react";
import Dashboard from "@/pages/Dashboard";
import Tasks from "@/pages/Tasks";
import Experiments from "@/pages/Experiments";
import NewTask from "@/pages/NewTask";
import Containers from "@/pages/Containers";

type TabId = "dashboard" | "tasks" | "experiments" | "new-task" | "containers";

interface MenuItem {
	id: TabId;
	name: string;
	component: React.ComponentType;
	icon: React.ReactNode;
	hideInMenu?: boolean;
}

interface MenuSection {
	title: string;
	items: MenuItem[];
}

const menuSections: MenuSection[] = [
	{
		title: "Overview",
		items: [
			{
				id: "dashboard",
				name: "Dashboard",
				component: Dashboard,
				icon: (
					<svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
						<path
							strokeLinecap="round"
							strokeLinejoin="round"
							strokeWidth={2}
							d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"
						/>
					</svg>
				)
			}
		]
	},
	{
		title: "Autotuning",
		items: [
			{
				id: "tasks",
				name: "Tasks",
				component: Tasks,
				icon: (
					<svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
						<path
							strokeLinecap="round"
							strokeLinejoin="round"
							strokeWidth={2}
							d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01"
						/>
					</svg>
				)
			},
			{
				id: "experiments",
				name: "Experiments",
				component: Experiments,
				icon: (
					<svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
						<path
							strokeLinecap="round"
							strokeLinejoin="round"
							strokeWidth={2}
							d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
						/>
					</svg>
				)
			},
			{
				id: "new-task",
				name: "New Task",
				component: NewTask,
				hideInMenu: true,
				icon: null as any
			}
		]
	},
	{
		title: "Infrastructure",
		items: [
			{
				id: "containers",
				name: "Containers",
				component: Containers,
				icon: (
					<svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
						<path
							strokeLinecap="round"
							strokeLinejoin="round"
							strokeWidth={2}
							d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
						/>
					</svg>
				)
			}
		]
	}
];

// Flatten menu items for lookup
const allMenuItems = menuSections.flatMap((section) => section.items);

// Simple navigation context to share state
export let navigateTo: (tabId: TabId) => void = () => {};

// Helper to get tab from URL hash
const getTabFromHash = (): TabId => {
	const hash = window.location.hash.slice(1); // Remove leading #
	const validTabs: TabId[] = ["dashboard", "tasks", "experiments", "new-task", "containers"];
	return validTabs.includes(hash as TabId) ? (hash as TabId) : "dashboard";
};

export default function Layout() {
	// Initialize activeTab from URL hash, or default to "dashboard"
	const [activeTab, setActiveTab] = useState<TabId>(getTabFromHash);
	const [sidebarOpen, setSidebarOpen] = useState(false);

	// Update URL hash when tab changes
	const updateActiveTab = (tabId: TabId) => {
		setActiveTab(tabId);
		window.location.hash = tabId;
	};

	// Expose navigation function
	navigateTo = (tabId: TabId) => updateActiveTab(tabId);

	// Listen for hash changes (browser back/forward navigation)
	useEffect(() => {
		const handleHashChange = () => {
			const tabFromHash = getTabFromHash();
			setActiveTab(tabFromHash);
		};

		window.addEventListener("hashchange", handleHashChange);
		return () => window.removeEventListener("hashchange", handleHashChange);
	}, []);

	const ActiveComponent =
		allMenuItems.find((item) => item.id === activeTab)?.component || Dashboard;

	return (
		<div className="h-screen flex overflow-hidden bg-gray-100">
			{/* Mobile sidebar backdrop */}
			{sidebarOpen && (
				<div
					className="fixed inset-0 bg-gray-600 bg-opacity-75 z-20 lg:hidden"
					onClick={() => setSidebarOpen(false)}
				></div>
			)}

			{/* Sidebar */}
			<div
				className={`fixed inset-y-0 left-0 w-64 bg-white border-r border-gray-200 transform transition-transform duration-300 ease-in-out z-30 ${
					sidebarOpen ? "translate-x-0" : "-translate-x-full"
				} lg:translate-x-0 lg:static lg:inset-0 flex flex-col`}
			>
				{/* Sidebar Header */}
				<div className="flex items-center justify-between h-16 px-4 border-b border-gray-200 flex-shrink-0">
					<div className="flex items-center space-x-3">
						<div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center">
							<span className="text-white text-sm font-bold">AI</span>
						</div>
						<div>
							<h1 className="text-sm font-bold text-gray-900">Inference Autotuner</h1>
							<p className="text-xs text-gray-500">v0.1.0</p>
						</div>
					</div>
					<button
						className="lg:hidden text-gray-400 hover:text-gray-600 p-1"
						onClick={() => setSidebarOpen(false)}
					>
						<svg
							className="w-5 h-5"
							fill="none"
							viewBox="0 0 24 24"
							stroke="currentColor"
						>
							<path
								strokeLinecap="round"
								strokeLinejoin="round"
								strokeWidth={2}
								d="M6 18L18 6M6 6l12 12"
							/>
						</svg>
					</button>
				</div>

				{/* Navigation Menu */}
				<nav className="flex-1 overflow-y-auto py-4 px-3">
					{menuSections.map((section, sectionIndex) => (
						<div key={sectionIndex} className={sectionIndex > 0 ? "mt-6" : ""}>
							{/* Section Title */}
							<div className="px-3 mb-2">
								<h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
									{section.title}
								</h3>
							</div>

							{/* Section Items */}
							<div className="space-y-1">
								{section.items.filter(item => !item.hideInMenu).map((item) => (
									<button
										key={item.id}
										onClick={() => {
											updateActiveTab(item.id);
											setSidebarOpen(false);
										}}
										className={`w-full flex items-center px-3 py-2 text-sm font-medium rounded-md transition-all duration-150 ${
											activeTab === item.id
												? "bg-blue-50 text-blue-700 shadow-sm"
												: "text-gray-700 hover:bg-gray-50 hover:text-gray-900"
										}`}
									>
										<span
											className={`flex-shrink-0 ${activeTab === item.id ? "text-blue-600" : "text-gray-400"}`}
										>
											{item.icon}
										</span>
										<span className="ml-3">{item.name}</span>
										{activeTab === item.id && (
											<span className="ml-auto">
												<svg
													className="w-4 h-4 text-blue-600"
													fill="currentColor"
													viewBox="0 0 20 20"
												>
													<path
														fillRule="evenodd"
														d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
														clipRule="evenodd"
													/>
												</svg>
											</span>
										)}
									</button>
								))}
							</div>
						</div>
					))}
				</nav>

				{/* Sidebar Footer */}
				<div className="flex-shrink-0 border-t border-gray-200 p-4">
					<div className="flex items-center">
						<div className="flex-shrink-0">
							<div className="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center">
								<svg
									className="w-5 h-5 text-gray-500"
									fill="none"
									viewBox="0 0 24 24"
									stroke="currentColor"
								>
									<path
										strokeLinecap="round"
										strokeLinejoin="round"
										strokeWidth={2}
										d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
									/>
								</svg>
							</div>
						</div>
						<div className="ml-3 flex-1">
							<p className="text-sm font-medium text-gray-700">Admin</p>
							<p className="text-xs text-gray-500">Administrator</p>
						</div>
						<button className="ml-2 text-gray-400 hover:text-gray-600">
							<svg
								className="w-5 h-5"
								fill="none"
								viewBox="0 0 24 24"
								stroke="currentColor"
							>
								<path
									strokeLinecap="round"
									strokeLinejoin="round"
									strokeWidth={2}
									d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
								/>
								<path
									strokeLinecap="round"
									strokeLinejoin="round"
									strokeWidth={2}
									d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
								/>
							</svg>
						</button>
					</div>
				</div>
			</div>

			{/* Main content area */}
			<div className="flex-1 flex flex-col overflow-hidden">
				{/* Top bar */}
				<header className="bg-white border-b border-gray-200 flex-shrink-0">
					<div className="px-4 sm:px-6 lg:px-8">
						<div className="flex justify-between items-center h-16">
							{/* Left side */}
							<div className="flex items-center">
								<button
									type="button"
									className="lg:hidden inline-flex items-center justify-center p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-blue-500 mr-2"
									onClick={() => setSidebarOpen(true)}
								>
									<svg
										className="h-6 w-6"
										fill="none"
										viewBox="0 0 24 24"
										stroke="currentColor"
									>
										<path
											strokeLinecap="round"
											strokeLinejoin="round"
											strokeWidth={2}
											d="M4 6h16M4 12h16M4 18h16"
										/>
									</svg>
								</button>
								<div>
									<h2 className="text-xl font-semibold text-gray-900">
										{allMenuItems.find((item) => item.id === activeTab)?.name ||
											"Dashboard"}
									</h2>
									<p className="text-sm text-gray-500">
										{
											menuSections.find((s) =>
												s.items.some((i) => i.id === activeTab)
											)?.title
										}
									</p>
								</div>
							</div>

							{/* Right side */}
							<div className="flex items-center space-x-2">
								<button className="p-2 text-gray-400 hover:text-gray-600 rounded-full hover:bg-gray-100">
									<svg
										className="w-5 h-5"
										fill="none"
										viewBox="0 0 24 24"
										stroke="currentColor"
									>
										<path
											strokeLinecap="round"
											strokeLinejoin="round"
											strokeWidth={2}
											d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
										/>
									</svg>
								</button>
								<button className="p-2 text-gray-400 hover:text-gray-600 rounded-full hover:bg-gray-100 relative">
									<svg
										className="w-5 h-5"
										fill="none"
										viewBox="0 0 24 24"
										stroke="currentColor"
									>
										<path
											strokeLinecap="round"
											strokeLinejoin="round"
											strokeWidth={2}
											d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"
										/>
									</svg>
									<span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full"></span>
								</button>
							</div>
						</div>
					</div>
				</header>

				{/* Page content */}
				<main className="flex-1 overflow-y-auto bg-gray-50">
					<div className="py-6">
						<div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
							<ActiveComponent />
						</div>
					</div>
				</main>
			</div>
		</div>
	);
}
