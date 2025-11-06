import { useQuery } from "@tanstack/react-query";
import apiClient from "@/services/api";
import type { Profile } from "@/types/api";

interface ProfileSelectorProps {
	selectedProfiles: string[];
	onSelectProfile: (profileName: string) => void;
	onDeselectProfile: (profileName: string) => void;
	multiSelect?: boolean;
}

export default function ProfileSelector({
	selectedProfiles,
	onSelectProfile,
	onDeselectProfile,
	multiSelect = false
}: ProfileSelectorProps) {
	const { data: profiles, isLoading, error } = useQuery({
		queryKey: ["profiles"],
		queryFn: () => apiClient.getProfiles()
	});

	if (isLoading) {
		return <div className="text-gray-500">Loading profiles...</div>;
	}

	if (error) {
		return <div className="text-red-500">Failed to load profiles</div>;
	}

	const handleProfileClick = (profileName: string) => {
		if (selectedProfiles.includes(profileName)) {
			onDeselectProfile(profileName);
		} else {
			if (!multiSelect) {
				// Clear other selections in single-select mode
				selectedProfiles.forEach(p => onDeselectProfile(p));
			}
			onSelectProfile(profileName);
		}
	};

	return (
		<div className="space-y-4">
			<div className="flex items-center justify-between">
				<h3 className="text-lg font-semibold">Select Configuration Profile</h3>
				{multiSelect && (
					<span className="text-sm text-gray-500">
						{selectedProfiles.length} selected
					</span>
				)}
			</div>

			<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
				{profiles?.map((profile: Profile) => {
					const isSelected = selectedProfiles.includes(profile.name);
					return (
						<div
							key={profile.name}
							onClick={() => handleProfileClick(profile.name)}
							className={`
								border rounded-lg p-4 cursor-pointer transition-all
								${isSelected
									? "border-blue-500 bg-blue-50 shadow-md"
									: "border-gray-200 hover:border-gray-300 hover:shadow-sm"
								}
							`}
						>
							<div className="flex items-start justify-between mb-2">
								<h4 className="font-semibold text-gray-900">{profile.name}</h4>
								{isSelected && (
									<svg
										className="w-5 h-5 text-blue-500"
										fill="currentColor"
										viewBox="0 0 20 20"
									>
										<path
											fillRule="evenodd"
											d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
											clipRule="evenodd"
										/>
									</svg>
								)}
							</div>

							<p className="text-sm text-gray-600 mb-3">{profile.description}</p>

							<div className="space-y-2">
								<div className="text-xs text-gray-500">
									<span className="font-medium">Use case:</span> {profile.use_case}
								</div>

								<div className="flex flex-wrap gap-1">
									{profile.tags.map((tag) => (
										<span
											key={tag}
											className="inline-block px-2 py-0.5 text-xs bg-gray-100 text-gray-700 rounded"
										>
											{tag}
										</span>
									))}
								</div>

								{profile.recommended_for.length > 0 && (
									<div className="text-xs text-green-600">
										<span className="font-medium">Recommended for:</span>{" "}
										{profile.recommended_for.join(", ")}
									</div>
								)}
							</div>
						</div>
					);
				})}
			</div>

			{!multiSelect && selectedProfiles.length === 0 && (
				<p className="text-sm text-gray-500 text-center">
					Click a profile to select it
				</p>
			)}
		</div>
	);
}
